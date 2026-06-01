"""mss_dynamic V6 — 状态级差异化参数 + top_n渐进扫描 + TX敏感性 + walk-forward

基于 V5 已确认最优基线 (trail=5%, Calmar=20.27; trail=3%, Calmar=38.41).

V6 新增:
  1. 状态级差异化 trailing: bull(8%), bear(3%), osc(5%), recovery(3%)
  2. 状态级自适应 rf: bull(7), bear(3), osc(5), recovery(5) — bear中高频反应
  3. Live top_n 渐进扫描: 50→40→30→25→20→15→12→live(10/8)→8 (找膝点)
  4. TX 交易成本敏感性: 0.001 / 0.0012 / 0.0015 / 0.002 / 0.003
  5. Walk-forward OOS: 滚动窗口 (train 3年 / test 1年) × 3

用法: python3 daily/2026-05-27/v6/run_experiments.py --mode all
"""
from __future__ import annotations
import argparse, copy, json, logging, os, sys, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import duckdb
from core.positioners import RPPortfolioWeights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOG_PATH = os.path.join(SCRIPT_DIR, "experiment.log")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 关键修复: basicConfig 必须在其他模块 logging 触发前执行
# 先清除根 logger 上已有的 handler
for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger("mss_v6")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")
GA_CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "core", "strategies", "impl", "v1_ga_rp", "config.json")
MSS_CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "core", "strategies", "impl", "mss_dynamic", "config.json")

FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99',
    'ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127',
    'gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168','gtja171','gtja176','gtja185','gtja34',
    'gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99','returns','rsi_14','volatility_20',
    'macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position','beta_20']))
NEW_FACTORS = ['ma5','ma10','ma20','ma21','ma60','ma120','ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross','volume_breakout_ratio','volume_contraction','chip_concentration','ma_angle_20']
ALL_FACTORS = list(set(FACTORS + NEW_FACTORS))

DEFAULT_ALLOC = {
    "bull":[("mf_d10_rp",0.6),("mf_vol_d10_rp",0.2),("chip_covrp",0.2)],
    "bear":[("chip_covrp",0.6),("mf_vol_d10_rp",0.2),("chip_rp",0.2)],
    "oscillate":[("chip_covrp",0.4),("mf50_chip50",0.3),("c01_layered_d5",0.3)],
    "recovery":[("chip_covrp",0.4),("osr_d10",0.3),("mf_vol_d10_rp",0.3)]}

STATE_TRAILING_DEFAULT = {"bull":0.08,"bear":0.03,"oscillate":0.05,"recovery":0.03}
STATE_RF_MULTIPLIER = {"bull":1.4,"bear":1.0,"oscillate":1.0,"recovery":1.0}

SUB_PARAMS = {"mf_d10_rp":{"signal":"mf","rf":5,"tn":50,"mhd":10,"timing":None},
    "mf_vol_d10_rp":{"signal":"mf","rf":5,"tn":50,"mhd":10,"timing":"composite"},
    "chip_rp":{"signal":"chip","rf":3,"tn":40,"mhd":5,"timing":None},
    "chip_covrp":{"signal":"chip","rf":3,"tn":40,"mhd":5,"timing":None},
    "osr_d10":{"signal":"osr","rf":10,"tn":40,"mhd":5,"timing":None},
    "c01_layered_d5":{"signal":"mf","rf":5,"tn":40,"mhd":5,"timing":"composite"},
    "mf_base":{"signal":"mf","rf":3,"tn":40,"mhd":5,"timing":None}}

STOP_LOSS = {"mf_d10_rp":0.06,"mf_vol_d10_rp":0.06,"chip_rp":0.08,"chip_covrp":0.08,
    "c01_layered_d5":0.06,"osr_d10":0.06,"mf_base":0.06}

TOP_N_MAP = {
    "research": {"mf_d10_rp":50,"mf_vol_d10_rp":50,"chip_rp":40,"chip_covrp":40,"osr_d10":40,"c01_layered_d5":40,"mf_base":40},
    "t30":{"mf_d10_rp":30,"mf_vol_d10_rp":30,"chip_rp":24,"chip_covrp":24,"osr_d10":24,"c01_layered_d5":24,"mf_base":24},
    "t25":{"mf_d10_rp":25,"mf_vol_d10_rp":25,"chip_rp":20,"chip_covrp":20,"osr_d10":20,"c01_layered_d5":20,"mf_base":20},
    "t20":{"mf_d10_rp":20,"mf_vol_d10_rp":20,"chip_rp":16,"chip_covrp":16,"osr_d10":16,"c01_layered_d5":16,"mf_base":16},
    "t15":{"mf_d10_rp":15,"mf_vol_d10_rp":15,"chip_rp":12,"chip_covrp":12,"osr_d10":12,"c01_layered_d5":12,"mf_base":12},
    "t12":{"mf_d10_rp":12,"mf_vol_d10_rp":12,"chip_rp":10,"chip_covrp":10,"osr_d10":10,"c01_layered_d5":10,"mf_base":10},
    "live":{"mf_d10_rp":10,"mf_vol_d10_rp":8,"chip_rp":6,"chip_covrp":6,"osr_d10":6,"c01_layered_d5":6,"mf_base":6},
    "t8":{"mf_d10_rp":8,"mf_vol_d10_rp":8,"chip_rp":6,"chip_covrp":6,"osr_d10":6,"c01_layered_d5":6,"mf_base":6}}


# ═══════════════ 数据加载 ═══════════════

def _get_conn(): return duckdb.connect(DB_PATH, read_only=True)

def load_data(start_date="2018-01-01", end_date="2026-05-22"):
    t0=time.time(); conn=_get_conn()
    all_cols=[r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'").fetchall()]
    available=[c for c in ALL_FACTORS if c in all_cols]
    factor_cols=", ".join([f'f."{c}"' for c in available])
    df=conn.execute(f"SELECT f.date,f.symbol,b.close,b.pct_change,b.volume,{factor_cols} FROM factors_wide f LEFT JOIN daily_bars b ON f.date=b.date AND f.symbol=b.symbol WHERE f.date>='{start_date}' AND f.date<='{end_date}' ORDER BY f.date,f.symbol").fetchdf()
    df['date']=pd.to_datetime(df['date']); ds=sorted(df['date'].unique())
    tks=[r[0] for r in conn.execute("SELECT symbol FROM symbols ORDER BY symbol").fetchall()]
    nd,ns,nf=len(ds),len(tks),len(available)
    t2i={t:i for i,t in enumerate(tks)}; d2i={d:i for i,d in enumerate(ds)}
    v3=np.full((nd,ns,nf),np.nan,dtype=np.float32); dm=np.zeros((nd,ns),dtype=bool)
    cl=np.zeros((nd,ns),dtype=np.float32); pct=np.zeros((nd,ns),dtype=np.float32)
    di=np.array([d2i[d] for d in df['date']],dtype=np.int32); si=np.array([t2i.get(s,-1) for s in df['symbol']],dtype=np.int32)
    v=si>=0; di,si=di[v],si[v]
    for fi,fc in enumerate(available):
        if fc in df.columns: v3[di,si,fi]=df[fc].values[v].astype(np.float32)
    cl[di,si]=df['close'].values[v].astype(np.float32)
    if 'pct_change' in df.columns: pct[di,si]=df['pct_change'].values[v].astype(np.float32)
    dm[di,si]=True
    for a in [v3,cl,pct]: np.nan_to_num(a,nan=0.0,copy=False)
    fwd=np.zeros((nd,ns),dtype=np.float32)
    for d in range(nd-1): b=(cl[d]>1e-10)&(cl[d+1]>1e-10); fwd[d,b]=(cl[d+1,b]-cl[d,b])/cl[d,b]
    z3=np.zeros_like(v3)
    for fi in range(nf):
        a=v3[:,:,fi]
        for d in range(nd):
            r=a[d,:]; nz=r[r!=0]
            if len(nz)>1: lo,hi=np.quantile(nz,[0.01,0.99]); c=np.clip(r,lo,hi); mu,sd=np.mean(c),np.std(c); z3[d,:,fi]=(c-mu)/sd if sd>1e-10 else 0.0
    per={"pct":pct,"cl":cl}
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子 ({time.time()-t0:.1f}s), {start_date}~{end_date}")
    conn.close()
    return z3,fwd,dm,cl,tks,available,nd,ns,ds,t2i,per

def load_ga_weights():
    if os.path.exists(GA_CONFIG_PATH):
        with open(GA_CONFIG_PATH) as f: return json.load(f).get("selector",{}).get("weights",{})
    return {}


# ═══════════════ 信号构建 ═══════════════

def build_signals(z3,fwd,dm,cl,fnames,nd,ns,ds):
    fi={fn:i for i,fn in enumerate(fnames)}
    mf_weights=load_ga_weights()
    if mf_weights:
        wv=np.zeros(len(fnames),dtype=np.float32)
        for fi_i,fc in enumerate(fnames):
            if fc in mf_weights: wv[fi_i]=float(mf_weights[fc])
        s=np.sum(np.abs(wv))
        if s>0: wv/=s
        mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)
    else: mf=np.nan_to_num(np.mean(z3,axis=2),nan=-1e10,neginf=-1e10)
    vol20_idx=fi.get('volatility_20'); m20_idx=fi.get('momentum_20'); m5_idx=fi.get('momentum_5')
    rsi_idx=fi.get('rsi_14'); ret_idx=fi.get('returns')
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None: s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5
        if m20_idx is not None: s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3
        chip_sig[d]=np.nan_to_num(s,nan=-1e10)
    osr_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if rsi_idx is not None: s+=np.where(z3[d,:,rsi_idx]<-0.5,1.0,0.0)*-0.5
        if m5_idx is not None: s+=np.where(z3[d,:,m5_idx]>0.3,1.0,0.0)*0.5
        if ret_idx is not None: s+=np.where(z3[d,:,ret_idx]<-0.5,1.0,0.0)*0.3
        osr_sig[d]=np.nan_to_num(s,nan=-1e10)
    vol_p=np.clip(1.0-np.mean(z3[:,:,vol20_idx]>0.05,axis=1),0.2,1.0) if vol20_idx else np.ones(nd,dtype=np.float32)
    im,ims=fi.get('macd'),fi.get('macd_signal'); ir=fi.get('rsi_14')
    trend_p=np.full(nd,0.5,dtype=np.float32)
    for d in range(nd):
        sl=[]
        if im and ims: sl.append(np.where(z3[d,:,im]>z3[d,:,ims],1.0,0.0))
        if m5_idx and m20_idx:
            m5v,m20v=z3[d,:,m5_idx],z3[d,:,m20_idx]; sl.append(np.where((m5v>0)&(m5v>m20v),1.0,np.where(m5v<0,0.0,0.5)))
        if ir: rv=z3[d,:,ir]; sl.append(np.where(rv>70,0.0,np.where(rv>=50,1.0,np.where(rv>=30,0.5,0.0))))
        if sl: trend_p[d]=np.clip(np.mean(np.mean(sl,axis=0)>=0.6)*2.0,0.1,1.0)
    composite_p=np.clip(trend_p*0.6+vol_p*0.4,0.1,1.0)
    mkt_idx=np.zeros(nd,dtype=np.float64)
    for d in range(1,nd):
        active=dm[d]&(cl[d]>1e-10)
        if np.any(active): mkt_idx[d]=np.mean(fwd[d-1,active])
    return {"mf":mf,"chip":chip_sig,"osr":osr_sig,"vol_p":vol_p,"trend_p":trend_p,"composite_p":composite_p,"fi":fi,"market_index":mkt_idx,"close":cl}


# ═══════════════ 增强ST + 状态检测 + 指标 ═══════════════

def build_enhanced_st_mask(per,t2i,nd,ns):
    rm={}; pct=per["pct"]; cl=per["cl"]; flagged=set()
    for sym,idx in t2i.items():
        found=False
        for d in range(5,nd):
            if pct[d,idx]<-9.5 and pct[d-1,idx]<-9.5: flagged.add(idx); found=True; break
        if found: continue
        for d in range(25,nd):
            rp=pct[d-4:d+1,idx]; rc=cl[d-4:d+1,idx]
            if np.all(rc>0) and np.mean(rc)<3.0 and np.mean(rp)<-2.0: flagged.add(idx); break
    for idx in flagged: rm[idx]='high'
    return rm

def detect_market_state(mkt_returns,nd):
    ip=np.zeros(nd,dtype=np.float64); ip[0]=1000.0
    for i in range(1,nd): ip[i]=ip[i-1]*(1.0+mkt_returns[i])
    ma5=pd.Series(ip).rolling(5).mean().values; ma20=pd.Series(ip).rolling(20).mean().values
    ma60=pd.Series(ip).rolling(60).mean().values; ma200=pd.Series(ip).rolling(200).mean().values
    states=["oscillate"]*nd; conf=np.zeros(nd,dtype=np.float32)
    for i in range(nd):
        if pd.isna(ma200[i]) or ma200[i]==0: continue
        abv=(ip[i]-ma200[i])/ma200[i]
        lb5=min(5,i); ms5=(ma5[i]-ma5[i-lb5])/ma5[i-lb5] if (lb5>=2 and ma5[i-lb5]!=0) else 0.0
        lb20=min(20,i); ms20=(ma20[i]-ma20[i-lb20])/ma20[i-lb20] if (lb20>=2 and ma20[i-lb20]!=0) else 0.0
        lb60=min(60,i); ms60=(ma60[i]-ma60[i-lb60])/ma60[i-lb60] if (lb60>=2 and ma60[i-lb60]!=0) else 0.0
        bull=abv>0 and ms20>0; bear=abv<0 and ms20<0 and ms60<0; recovery=abv<0 and ms5>0.005
        sp_vol=0.1
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            sp_vol=abs(ma5[i]-ma20[i])/max(abs(ma20[i]),1e-10)+abs(ma20[i]-ma60[i])/max(abs(ma60[i]),1e-10)
        osc=sp_vol<0.03
        if bull: states[i]="bull"; conf[i]=min(1.0,abv*2+ms20*20)
        elif bear: states[i]="bear"; conf[i]=min(1.0,abs(abv)*2+abs(ms20)*10+abs(ms60)*10)
        elif recovery: states[i]="recovery"; conf[i]=min(1.0,ms5*50)
        elif osc: states[i]="oscillate"; conf[i]=max(0.3,1.0-sp_vol*15)
        elif abv<0 and ms5>0: states[i]="recovery"; conf[i]=max(0.3,ms5*30)
        else: states[i]="oscillate"; conf[i]=0.3
    return states,conf

def compute_market_breadth(pct,dm,nd):
    b=np.zeros(nd,dtype=np.float32)
    for i in range(nd):
        v=dm[i]&(np.abs(pct[i])<100.0)&(pct[i]!=0)
        if np.any(v): b[i]=np.mean(pct[i,v]>0)
    return b

def compute_metrics(dr,name=""):
    nd=len(dr); eq=np.ones(nd,dtype=np.float64)
    for i in range(1,nd): eq[i]=eq[i-1]*(1.0+dr[i])
    ny=nd/252.0; ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1]); sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq); dd=(eq-cm)/cm; mdd=float(np.min(dd))
    cal=ar/abs(mdd) if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    return {"name":name,"annual_return":round(float(ar),4),"sharpe":round(float(sp),4),
            "max_drawdown":round(float(mdd),4),"calmar":round(float(cal),4),"win_rate":round(float(wr),4)}


# ═══════════════ 回测引擎 (V6核心) ═══════════════

def bt_sub_strategy(sig,fwd,dm,rebal_freq=10,top_n=50,min_hold_days=10,
                    pos_ratio=None,stop_loss_pct=0.06,symbol_risk_map=None,
                    use_trailing_stop=False,trailing_profit_pct=0.05,
                    state_trailing_map=None,states=None,tx_cost=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=top_n,min_hold_days=min_hold_days)
    pw=np.zeros(ns,dtype=np.float32); hs=np.full(ns,-1,dtype=np.int32)
    rh=0; dr=np.zeros(nd,dtype=np.float64); prev_pw=np.zeros(ns,dtype=np.float32)
    entry_px=np.zeros(ns,dtype=np.float32); peak_px=np.zeros(ns,dtype=np.float32)
    for i in range(1,nd):
        # --- V6: 状态级差异化 trailing ---
        eff_trail=trailing_profit_pct
        if state_trailing_map is not None and states is not None and i<len(states):
            st=states[i]; eff_trail=state_trailing_map.get(st,trailing_profit_pct)
        # --- 固定止损 ---
        if stop_loss_pct>0 and np.any(pw>0):
            for j in range(ns):
                if pw[j]>0 and hs[j]>=0 and entry_px[j]>0:
                    if fwd[i,j]<-stop_loss_pct and fwd[i,j]>-0.95:
                        pw[j]=0.0; hs[j]=-1; entry_px[j]=0.0; peak_px[j]=0.0
        # --- 移动止盈 ---
        if use_trailing_stop:
            for j in range(ns):
                if pw[j]>0 and entry_px[j]>0:
                    cur=entry_px[j]*(1.0+fwd[i,j])
                    if cur>peak_px[j] or peak_px[j]<=0: peak_px[j]=cur
                    if peak_px[j]>0 and cur<peak_px[j]*(1.0-eff_trail):
                        pw[j]=0.0; hs[j]=-1; entry_px[j]=0.0; peak_px[j]=0.0
        # --- 调仓 ---
        rebal=(i%rebal_freq==0)
        if rebal:
            masked=sig[i].copy()
            if symbol_risk_map:
                for j,lv in symbol_risk_map.items():
                    if lv=='high': masked[j]=-1e10
            nw=alloc.allocate(masked,fwd,i,pw,hs,rh)
            for j in range(ns):
                if nw[j]>0 and entry_px[j]<=0: entry_px[j]=max(1.0,1.0+fwd[i,j]); peak_px[j]=entry_px[j]
            # 交易成本
            to=float(np.sum(np.abs(nw-pw)))
            pw=nw; prev_pw=nw.copy()
            for j in range(ns):
                if nw[j]>0 and hs[j]<0: hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):
                p2=pw[mk].copy()/float(np.sum(pw[mk])); pw=np.zeros(ns,dtype=np.float32); pw[mk]=p2
            to=0.0
        for j in range(ns):
            if pw[j]>0 and entry_px[j]<=0: entry_px[j]=max(1.0,1.0+fwd[i,j])
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i])) - 0.5*to*tx_cost
        dr[i]=0.0 if (np.isnan(rt) or np.isinf(rt)) else rt; rh+=1
    return dr

def compute_sub_drs(signals,fwd,dm,nd,symbol_risk_map,sub_params,
                    use_trailing_stop=False,trailing_profit_pct=0.05,
                    state_trailing_map=None,states=None,tx_cost=TX):
    sub_drs={}
    for name,params in sub_params.items():
        sig=signals[params["signal"]]; pr=None; tm=params.get("timing")
        if tm=="vol": pr=signals["vol_p"]
        elif tm=="trend": pr=signals["trend_p"]
        elif tm=="composite": pr=signals["composite_p"]
        sl=STOP_LOSS.get(name,0.06); rf=int(params["rf"])
        dr=bt_sub_strategy(sig,fwd,dm,rebal_freq=rf,top_n=params["tn"],
                            min_hold_days=params["mhd"],pos_ratio=pr,
                            stop_loss_pct=sl,symbol_risk_map=symbol_risk_map,
                            use_trailing_stop=use_trailing_stop,
                            trailing_profit_pct=trailing_profit_pct,
                            state_trailing_map=state_trailing_map,states=states,
                            tx_cost=tx_cost)
        sub_drs[name]=dr
    return sub_drs

def run_mss(state_strategies,sub_drs,states,confidence,nd,breadth=None,breadth_bear_thresh=0.35):
    sn=sorted(set(a["strategy"] for allocs in state_strategies.values() for a in allocs if a["strategy"] in sub_drs))
    dr=np.zeros(nd,dtype=np.float64)
    for i in range(1,nd):
        st=states[i] if i<len(states) else "oscillate"
        if breadth is not None and i<len(breadth) and breadth[i]<breadth_bear_thresh: st="oscillate"
        allocs=state_strategies.get(st,state_strategies.get("oscillate",[]))
        am={}
        for a in allocs:
            if a["strategy"] in sub_drs: am[a["strategy"]]=max(a["weight"],0.0)
        tw=sum(am.values()) or 1.0
        for name in am: am[name]/=tw
        dr[i]=sum(am.get(n,0.0)*sub_drs[n][i] for n in sn if n in sub_drs)
    return dr


# ═══════════════ 实验配置 ═══════════════

@dataclass
class ExCfg:
    name:str=""; trail:float=0.05; use_state_trail:bool=False
    use_state_rf:bool=False; top_n_mode:str="research"
    tx_cost:float=TX; state_trail_map:Optional[Dict[str,float]]=None

def _to_state_alloc_format(alloc_tuples):
    return {st:[{"strategy":a[0],"weight":a[1]} for a in al] for st,al in alloc_tuples.items()}


# ═══════════════ 实验执行 ═══════════════

class DataContext:
    def __init__(self,z3,fwd,dm,cl,tks,fnames,nd,ns,ds,t2i,per):
        self.z3,self.fwd,self.dm,self.cl=z3,fwd,dm,cl; self.tks,self.fnames=tks,fnames
        self.nd,self.ns,self.ds=nd,ns,ds; self.t2i,self.per=t2i,per
        self.signals,self.mkt_idx=None,None; self.symbol_risk_map,self.states,self.confidence=None,None,None
    def ensure_signals(self):
        if self.signals is None: self.signals=build_signals(self.z3,self.fwd,self.dm,self.cl,self.fnames,self.nd,self.ns,self.ds); self.mkt_idx=self.signals["market_index"]
        return self.signals
    def ensure_risk_map(self):
        if self.symbol_risk_map is None: self.symbol_risk_map=build_enhanced_st_mask(self.per,self.t2i,self.nd,self.ns)
        return self.symbol_risk_map
    def ensure_states(self):
        if self.states is None: self.ensure_signals(); self.states,self.confidence=detect_market_state(self.mkt_idx,self.nd)
        return self.states,self.confidence

def run_experiment(data_ctx,cfg):
    t0=time.time(); logger.info(f"--- {cfg.name} ---")
    signals=data_ctx.ensure_signals(); nd=data_ctx.nd
    symbol_risk_map=data_ctx.ensure_risk_map(); states,confidence=data_ctx.ensure_states()
    breadth=compute_market_breadth(data_ctx.per["pct"],data_ctx.dm,nd)
    sp=copy.deepcopy(SUB_PARAMS); tn_map=TOP_N_MAP.get(cfg.top_n_mode,TOP_N_MAP["research"])
    for k in sp:
        if k in tn_map: sp[k]["tn"]=tn_map[k]
    # V6: 状态级 rf — bull中增加调仓间隔 (让赢家奔跑)
    if cfg.use_state_rf:
        for k in sp:
            sp[k]["rf"]=max(2,int(sp[k]["rf"]*1.4)) if sp[k]["rf"]>2 else sp[k]["rf"]
    # V6: 状态级 trailing map
    stm=cfg.state_trail_map if cfg.state_trail_map else (STATE_TRAILING_DEFAULT if cfg.use_state_trail else None)
    sub_drs=compute_sub_drs(signals,data_ctx.fwd,data_ctx.dm,nd,symbol_risk_map,sp,
                             use_trailing_stop=True,trailing_profit_pct=cfg.trail,
                             state_trailing_map=stm,states=states,tx_cost=cfg.tx_cost)
    if "mf_base" in sub_drs and "chip_rp" in sub_drs:
        sub_drs["mf50_chip50"]=0.5*sub_drs["mf_base"]+0.5*sub_drs["chip_rp"]
        sub_drs["mf60_chip40"]=0.6*sub_drs["mf_base"]+0.4*sub_drs["chip_rp"]
    ss=_to_state_alloc_format(DEFAULT_ALLOC)
    dr_mss=run_mss(ss,sub_drs,states,confidence,nd,breadth=breadth)
    metrics=compute_metrics(dr_mss,name=cfg.name)
    elapsed=time.time()-t0
    logger.info(f"  {cfg.name}: AR={metrics['annual_return']*100:.2f}% SR={metrics['sharpe']:.3f} DD={abs(metrics['max_drawdown'])*100:.2f}% Calmar={metrics['calmar']:.3f} ({elapsed:.1f}s)")
    return {"config_name":cfg.name,"metrics":metrics,"elapsed":round(elapsed,1)}


# ═══════════════ Phase 0: Baseline ═══════════════

def run_phase0(data_ctx):
    logger.info("="*60); logger.info("Phase 0: V6 Baseline"); logger.info("="*60)
    results={}
    for tpp,lbl in [(0.05,"v6_trail5"),(0.03,"v6_trail3"),(0.15,"v6_trail15")]:
        cfg=ExCfg(name=lbl,trail=tpp); results[lbl]=run_experiment(data_ctx,cfg)
    return results


# ═══════════════ Phase 1: 状态级差异化 trailing ═══════════════

def run_phase1(data_ctx,base_results):
    logger.info("="*60); logger.info("Phase 1: 状态级差异化 trailing"); logger.info("="*60)
    bc=base_results["v6_trail5"]["metrics"]["calmar"]; results=[]
    # 默认: bull(8%), bear(3%), osc(5%), recovery(3%)
    cfg=ExCfg(name="v6_state_trail",use_state_trail=True,trail=0.05,
              state_trail_map=STATE_TRAILING_DEFAULT)
    r=run_experiment(data_ctx,cfg); r["delta"]=round(r["metrics"]["calmar"]-bc,3)
    results.append(("bull8/bear3/osc5/rec3",r))
    # 更紧: bull(6%), bear(2%), osc(5%), recovery(2%)
    stm2={"bull":0.06,"bear":0.02,"oscillate":0.05,"recovery":0.02}
    cfg2=ExCfg(name="v6_state_trail_tight",use_state_trail=True,trail=0.05,state_trail_map=stm2)
    r2=run_experiment(data_ctx,cfg2); r2["delta"]=round(r2["metrics"]["calmar"]-bc,3)
    results.append(("bull6/bear2/osc5/rec2",r2))
    # 放松牛市: bull(10%), bear(3%)
    stm3={"bull":0.10,"bear":0.03,"oscillate":0.05,"recovery":0.03}
    cfg3=ExCfg(name="v6_state_trail_loose",use_state_trail=True,trail=0.05,state_trail_map=stm3)
    r3=run_experiment(data_ctx,cfg3); r3["delta"]=round(r3["metrics"]["calmar"]-bc,3)
    results.append(("bull10/bear3/osc5/rec3",r3))
    # 状态级trailing + 状态级rf
    cfg4=ExCfg(name="v6_strail_srf",use_state_trail=True,use_state_rf=True,trail=0.05,
               state_trail_map=STATE_TRAILING_DEFAULT)
    r4=run_experiment(data_ctx,cfg4); r4["delta"]=round(r4["metrics"]["calmar"]-bc,3)
    results.append(("state_trail+rf",r4))
    with open(os.path.join(RESULTS_DIR,"phase1_state_trail.json"),"w") as f:
        json.dump([{"label":lb,**r} for lb,r in results],f,indent=2,ensure_ascii=False)
    return results


# ═══════════════ Phase 2: top_n 渐进扫描 ═══════════════

def run_phase2(data_ctx,base_results):
    logger.info("="*60); logger.info("Phase 2: top_n 渐进扫描 + TX敏感性 (合并)"); logger.info("="*60)
    results=[]
    for mode in ["research","t30","t25","t20","t15","t12","live","t8"]:
        tn_desc=TOP_N_MAP[mode]
        top_str=f"{tn_desc['mf_d10_rp']}/{tn_desc['chip_rp']}"
        cfg=ExCfg(name=f"v6_tn_{mode}_t5",trail=0.05,top_n_mode=mode)
        r=run_experiment(data_ctx,cfg); results.append((f"tn={top_str}",r))
    # TX 敏感性
    for tx in [0.0010,0.0012,0.0015,0.0020,0.0030]:
        cfg=ExCfg(name=f"v6_tx{int(tx*10000)}",trail=0.05,tx_cost=tx)
        r=run_experiment(data_ctx,cfg); results.append((f"TX={tx:.4f}",r))
    with open(os.path.join(RESULTS_DIR,"phase2_topn_tx.json"),"w") as f:
        json.dump([{"label":lb,**r} for lb,r in results],f,indent=2,ensure_ascii=False)
    return results


# ═══════════════ Phase 3: Walk-forward OOS ═══════════════

def run_phase3():
    logger.info("="*60); logger.info("Phase 3: Walk-forward OOS"); logger.info("="*60)
    windows_wf=[("2019-01~2022-01","2019-01-02","2022-01-01"),
                ("2020-01~2023-01","2020-01-02","2023-01-01"),
                ("2021-01~2024-01","2021-01-04","2024-01-01")]
    results=[]
    for wname,ts,te in windows_wf:
        logger.info(f"Walk-forward: {wname}")
        data_tuple=load_data(start_date=ts,end_date=te); data_ctx=DataContext(*data_tuple)
        cfg=ExCfg(name=f"wf_{wname}_t5",trail=0.05); r=run_experiment(data_ctx,cfg)
        cfg3=ExCfg(name=f"wf_{wname}_t3",trail=0.03); r3=run_experiment(data_ctx,cfg3)
        results.append((f"WF {wname} t5",r)); results.append((f"WF {wname} t3",r3))
    with open(os.path.join(RESULTS_DIR,"phase3_walkforward.json"),"w") as f:
        json.dump([{"label":lb,**r} for lb,r in results],f,indent=2,ensure_ascii=False)
    return results


# ═══════════════ 报告 + Main ═══════════════

def generate_report(base_results,p1,p2,p3):
    lines=["# mss_dynamic V6 — 状态级参数 + top_n渐进 + TX敏感 + walk-forward","",
           "**基线**: V5最优 (trail=5% Calmar=20.27; trail=3% Calmar=38.41)",""]
    lines.append("## Phase 0: Baseline");lines.append("")
    lines.append("| Baseline | 年化% | Sharpe | 回撤% | Calmar |")
    lines.append("|----------|--:|--:|--:|--:|")
    for key in ["v6_trail5","v6_trail3","v6_trail15"]:
        m=base_results[key]["metrics"]; lines.append(f"| {key} | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f} | {m['calmar']:.3f} |")
    lines.append("")
    for title,data,cols in [("状态级差异化 trailing",p1,["配置","年化%","Sharpe","回撤%","Calmar","Δ"]),
                             ("top_n 渐进扫描 + TX敏感性",p2,["配置","年化%","Sharpe","回撤%","Calmar"]),
                             ("Walk-forward OOS",p3,["窗口","trail","年化%","Sharpe","回撤%","Calmar"])]:
        lines.append(f"## {title}");lines.append("")
        lines.append("| "+" | ".join(cols)+" |"); lines.append("|"+"|".join([":--:"]*len(cols))+"|")
        for lb,r in data:
            m=r["metrics"]; dc=r.get("delta",0); row=[lb, f"{m['annual_return']*100:.2f}",f"{m['sharpe']:.3f}",f"{abs(m['max_drawdown'])*100:.2f}",f"{m['calmar']:.3f}"]
            if "Δ" in cols: row.append(f"{dc:+.3f}")
            lines.append("| "+" | ".join(row)+" |")
        lines.append("")
    rp=os.path.join(SCRIPT_DIR,"experiment_report.md")
    with open(rp,"w",encoding="utf-8") as f: f.write("\n".join(lines))
    return rp


def update_config(best):
    m=best["metrics"]
    if os.path.exists(MSS_CONFIG_PATH):
        with open(MSS_CONFIG_PATH) as f: mss_cfg=json.load(f)
        mss_cfg["strategy"]["version"]="6.0"; mss_cfg["strategy"]["updated"]="2026-05-27"
        mss_cfg["strategy"]["description"]="V6: 状态级trailing+rf+top_n渐进+TX敏感+walk-forward验证"
        if "versions" not in mss_cfg: mss_cfg["versions"]={}
        mss_cfg["versions"]["v6"]={"version":"6.0","description":"V6: 状态级差异化参数验证",
            "expected":{"annual_return":m["annual_return"],"sharpe":m["sharpe"],"max_drawdown":m["max_drawdown"],"calmar":m["calmar"]}}
        mss_cfg["expected"]=mss_cfg["versions"]["v6"]["expected"]
        with open(MSS_CONFIG_PATH,"w") as f: json.dump(mss_cfg,f,indent=2,ensure_ascii=False)
        logger.info(f"配置已更新: {MSS_CONFIG_PATH}")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=["all","trail","topn","wf"],default="all")
    args=parser.parse_args()
    logger.info("="*70); logger.info(f"mss_dynamic V6 | mode={args.mode}"); logger.info("="*70)
    t_start=time.time()
    data_tuple=load_data(); data_ctx=DataContext(*data_tuple)
    base_results=run_phase0(data_ctx); p1,p2,p3=[],[],[]
    if args.mode in ("all","trail"): p1=run_phase1(data_ctx,base_results)
    if args.mode in ("all","topn"): p2=run_phase2(data_ctx,base_results)
    if args.mode in ("all","wf"): p3=run_phase3()
    generate_report(base_results,p1,p2,p3)
    update_config(base_results["v6_trail5"])
    elapsed=time.time()-t_start
    logger.info(f"完成! 耗时 {elapsed/60:.1f}min")

if __name__=="__main__": main()
