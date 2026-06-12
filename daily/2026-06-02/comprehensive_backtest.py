"""mss_dynamic 全量回测 + 归因分析 (最新数据至 2026-06-02)

基于 V6 框架 + 实盘口径参数，输出：
  1. 主策略全区间 + 分年窗口表现
  2. 各子策略独立回测 + 窗口表现
  3. 各市场状态下子策略贡献归因
  4. Walk-forward OOS 验证
  5. 子策略收益相关性分析
  6. 月度收益热力图 + 逐年收益

用法: python3 daily/2026-06-02/comprehensive_backtest.py
"""
from __future__ import annotations
import argparse, copy, json, logging, os, sys, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb
from core.positioners import RPPortfolioWeights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(SCRIPT_DIR, "backtest.log"), mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger("mss_bt")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")
GA_CONFIG_PATH = os.path.abspath("./core/strategies/impl/v1_ga_rp/config.json")

FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99',
    'ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127',
    'gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168','gtja171','gtja176','gtja185','gtja34',
    'gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99','returns','rsi_14','volatility_20',
    'macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position','beta_20']))
ALL_FACTORS = list(set(FACTORS))

WINDOWS = [
    ("2019修复牛", "2019-01-02", "2019-12-31"),
    ("2020疫情",   "2020-01-02", "2020-12-31"),
    ("2021结构牛", "2021-01-04", "2021-12-31"),
    ("2022熊市",   "2022-01-04", "2022-12-30"),
    ("2023震荡",   "2023-01-03", "2023-12-29"),
    ("2024反弹",   "2024-01-02", "2024-12-31"),
    ("2025至今",   "2025-01-02", "2026-06-02"),
]

DEFAULT_ALLOC = {
    "bull":[("mf_d10_rp",0.6),("mf_vol_d10_rp",0.2),("chip_covrp",0.2)],
    "bear":[("chip_covrp",0.6),("chip_equal_d3",0.2),("mf_vol_d10_rp",0.2)],
    "oscillate":[("chip_covrp",0.4),("mf50_chip50",0.3),("c01_layered_d5",0.3)],
    "recovery":[("chip_covrp",0.4),("osr_d10",0.3),("mf_vol_d10_rp",0.3)]}

SUB_PARAMS = {"mf_d10_rp":{"signal":"mf","rf":5,"tn":10,"mhd":10,"timing":None},
    "mf_vol_d10_rp":{"signal":"mf","rf":5,"tn":8,"mhd":10,"timing":"composite"},
    "chip_covrp":{"signal":"chip","rf":3,"tn":6,"mhd":5,"timing":None},
    "chip_equal_d3":{"signal":"chip","rf":3,"tn":6,"mhd":5,"timing":None},
    "chip_rp":{"signal":"chip","rf":3,"tn":6,"mhd":5,"timing":None},
    "osr_d10":{"signal":"osr","rf":10,"tn":6,"mhd":5,"timing":None},
    "c01_layered_d5":{"signal":"mf","rf":5,"tn":6,"mhd":5,"timing":"composite"},
    "mf_base":{"signal":"mf","rf":3,"tn":40,"mhd":5,"timing":None}}

STOP_LOSS = {"mf_d10_rp":0.06,"mf_vol_d10_rp":0.06,"chip_covrp":0.08,"chip_equal_d3":0.08,
    "c01_layered_d5":0.06,"osr_d10":0.06,"chip_rp":0.08,"mf_base":0.06}

RESEARCH_TOP_N = {"mf_d10_rp":50,"mf_vol_d10_rp":50,"chip_covrp":40,"chip_equal_d3":40,
    "chip_rp":40,"osr_d10":40,"c01_layered_d5":40,"mf_base":40}

LIVE_TOP_N = {"mf_d10_rp":10,"mf_vol_d10_rp":8,"chip_covrp":6,"chip_equal_d3":6,
    "chip_rp":6,"osr_d10":6,"c01_layered_d5":6,"mf_base":6}

# ═══════════════ 数据加载 ═══════════════

def _get_conn(): return duckdb.connect(DB_PATH, read_only=True)

def load_data(start_date="2018-01-01", end_date="2026-06-02"):
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
    total_ret=float(eq[-1]/eq[0]-1.0)
    ny=nd/252.0; ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1]); sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq); dd=(eq-cm)/cm; mdd=float(np.min(dd))
    cal=ar/abs(mdd) if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    # 回撤持续/修复天数
    dd_dur=0; rec_days=0
    if mdd<-1e-10:
        valley=int(np.argmin(dd))
        peak_before=np.max(eq[:valley+1])
        peak_idx=int(np.argmax(eq[:valley+1]))
        dd_dur=valley-peak_idx
        after=eq[valley+1:]
        hits=np.where(after>=peak_before)[0]
        rec_days=int(hits[0])+1 if len(hits)>0 else len(after)
    return {"name":name,"total_return":round(total_ret,4),"annual_return":round(float(ar),4),
            "sharpe":round(float(sp),4),"max_drawdown":round(float(mdd),4),"calmar":round(float(cal),4),
            "win_rate":round(float(wr),4),"dd_duration":dd_dur,"recovery_days":rec_days}

# ═══════════════ 回测引擎 ═══════════════

def bt_sub_strategy(sig,fwd,dm,rebal_freq=10,top_n=50,min_hold_days=10,
                    pos_ratio=None,stop_loss_pct=0.06,symbol_risk_map=None,
                    use_trailing_stop=False,trailing_profit_pct=0.05,
                    state_trailing_map=None,states=None,tx_cost=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=top_n,min_hold_days=min_hold_days)
    pw=np.zeros(ns,dtype=np.float32); hs=np.full(ns,-1,dtype=np.int32)
    rh=0; dr=np.zeros(nd,dtype=np.float64)
    entry_px=np.zeros(ns,dtype=np.float32); peak_px=np.zeros(ns,dtype=np.float32)
    for i in range(1,nd):
        eff_trail=trailing_profit_pct
        if state_trailing_map is not None and states is not None and i<len(states):
            st=states[i]; eff_trail=state_trailing_map.get(st,trailing_profit_pct)
        if stop_loss_pct>0 and np.any(pw>0):
            for j in range(ns):
                if pw[j]>0 and hs[j]>=0 and entry_px[j]>0:
                    if fwd[i,j]<-stop_loss_pct and fwd[i,j]>-0.95:
                        pw[j]=0.0; hs[j]=-1; entry_px[j]=0.0; peak_px[j]=0.0
        if use_trailing_stop:
            for j in range(ns):
                if pw[j]>0 and entry_px[j]>0:
                    cur=entry_px[j]*(1.0+fwd[i,j])
                    if cur>peak_px[j] or peak_px[j]<=0: peak_px[j]=cur
                    if peak_px[j]>0 and cur<peak_px[j]*(1.0-eff_trail):
                        pw[j]=0.0; hs[j]=-1; entry_px[j]=0.0; peak_px[j]=0.0
        rebal=(i%rebal_freq==0)
        if rebal:
            masked=sig[i].copy()
            if symbol_risk_map:
                for j,lv in symbol_risk_map.items():
                    if lv=='high': masked[j]=-1e10
            nw=alloc.allocate(masked,fwd,i,pw,hs,rh)
            for j in range(ns):
                if nw[j]>0 and entry_px[j]<=0: entry_px[j]=max(1.0,1.0+fwd[i,j]); peak_px[j]=entry_px[j]
            to=float(np.sum(np.abs(nw-pw)))
            pw=nw
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
        dr[i]=sum(am.get(n,0.0)*sub_drs[n][i] for n in am if n in sub_drs)
    return dr

# ═══════════════ 窗口分析 ═══════════════

def window_analysis(dr,ds,name,wnames=None):
    if wnames is None: wnames=WINDOWS
    nd=len(dr)
    total=compute_metrics(dr,f"{name}_全区间")
    results=[total]
    for wname,ws,we in wnames:
        ws_d=pd.Timestamp(ws); we_d=pd.Timestamp(we)
        idx=[j for j,d in enumerate(ds) if ws_d<=d<=we_d]
        if len(idx)<5:
            results.append({"name":name,"window":wname,"n_days":len(idx)}); continue
        w_dr=dr[idx[0]:idx[-1]+1].copy(); w_dr[0]=0.0
        wr=compute_metrics(w_dr,f"{name}_{wname}")
        results.append(wr)
    return results

def _to_state_alloc_format(alloc_tuples):
    return {st:[{"strategy":a[0],"weight":a[1]} for a in al] for st,al in alloc_tuples.items()}

# ═══════════════ 归因分析 ═══════════════

def attribution_by_state(sub_drs,states,nd):
    """各市场状态下子策略的收益贡献分布。"""
    attr={}
    for i in range(1,nd):
        st=states[i] if i<len(states) else "oscillate"
        if st not in attr: attr[st]={n:[] for n in sub_drs}
        for n in sub_drs: attr[st][n].append(sub_drs[n][i])
    result={}
    for st,data in attr.items():
        result[st]={}
        for n,rets in data.items():
            result[st][n]={"mean_ret":round(float(np.mean(rets)),6),"std_ret":round(float(np.std(rets)),6),
                           "win_rate":round(float(np.sum(np.array(rets)>0)/max(len(rets),1)),4),"n_days":len(rets)}
    return result

def attribution_sub_contribution(sub_drs,sds,states,alloc,nd):
    """各子策略对主策略的贡献分解。"""
    dr={n:np.zeros(nd,dtype=np.float64) for n in sds}
    dr_mss=np.zeros(nd,dtype=np.float64)
    for i in range(1,nd):
        st=states[i] if i<len(states) else "oscillate"
        allocs=alloc.get(st,alloc.get("oscillate",[]))
        am={}
        for a in allocs:
            if a["strategy"] in sds: am[a["strategy"]]=max(a["weight"],0.0)
        tw=sum(am.values()) or 1.0
        for n in am: am[n]/=tw
        for n in sds:
            w=am.get(n,0.0)
            dr[n][i]=w * sub_drs[n][i]
        dr_mss[i]=sum(dr[n][i] for n in sds)
    result={}
    for n in sds:
        c=compute_metrics(dr[n],n)
        result[n]=c
    result["mss_total"]=compute_metrics(dr_mss,"mss")
    return result

def correlation_analysis(sub_drs):
    """子策略收益相关性矩阵。"""
    names=list(sub_drs.keys())
    mat=np.zeros((len(names),len(names)),dtype=np.float32)
    for i in range(len(names)):
        for j in range(len(names)):
            a=sub_drs[names[i]]; b=sub_drs[names[j]]
            mask=(a!=0)&(b!=0)
            if np.sum(mask)>2:
                mat[i][j]=np.corrcoef(a[mask],b[mask])[0,1]
    return {"names":names,"correlation":mat.tolist()}

def monthly_returns(dr,ds,name):
    """计算月度收益矩阵。"""
    s=pd.Series(dr,index=pd.to_datetime(ds))
    m=s.groupby([s.index.year,s.index.month]).sum()
    m=m.reset_index(); m.columns=["year","month","return"]
    return m.groupby("year").apply(lambda g: dict(zip(g["month"],g["return"]))).to_dict()

def yearly_breakdown(dr,ds):
    """逐年收益分解。"""
    s=pd.Series(dr,index=pd.to_datetime(ds))
    y=s.groupby(s.index.year).sum()
    return {int(k):round(float(v),4) for k,v in y.items() if k>=2019}

# ═══════════════ 主回测逻辑 ═══════════════

@dataclass
class ExCfg:
    name:str=""; trail:float=0.05; use_state_trail:bool=False
    top_n_mode:str="live"; tx_cost:float=TX
    state_trail_map:Optional[Dict[str,float]]=None

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

def run_experiment(data_ctx,cfg,sub_params=None):
    t0=time.time(); logger.info(f"--- {cfg.name} ---")
    signals=data_ctx.ensure_signals(); nd=data_ctx.nd
    symbol_risk_map=data_ctx.ensure_risk_map(); states,confidence=data_ctx.ensure_states()
    breadth=compute_market_breadth(data_ctx.per["pct"],data_ctx.dm,nd)
    sp=copy.deepcopy(sub_params or SUB_PARAMS)
    tn_map={"live":LIVE_TOP_N,"research":RESEARCH_TOP_N}
    tn=tn_map.get(cfg.top_n_mode,LIVE_TOP_N)
    for k in sp:
        if k in tn: sp[k]["tn"]=tn[k]
    sub_drs=compute_sub_drs(signals,data_ctx.fwd,data_ctx.dm,nd,symbol_risk_map,sp,
                             use_trailing_stop=True,trailing_profit_pct=cfg.trail,
                             state_trailing_map=cfg.state_trail_map,states=states,
                             tx_cost=cfg.tx_cost)
    if "mf_base" in sub_drs and "chip_rp" in sub_drs:
        sub_drs["mf50_chip50"]=0.5*sub_drs["mf_base"]+0.5*sub_drs["chip_rp"]
        sub_drs["mf60_chip40"]=0.6*sub_drs["mf_base"]+0.4*sub_drs["chip_rp"]
    ss=_to_state_alloc_format(DEFAULT_ALLOC)
    dr_mss=run_mss(ss,sub_drs,states,confidence,nd,breadth=breadth)
    metrics=compute_metrics(dr_mss,name=cfg.name)
    elapsed=time.time()-t0
    logger.info(f"  {cfg.name}: AR={metrics['annual_return']*100:.2f}% SR={metrics['sharpe']:.3f} DD={abs(metrics['max_drawdown'])*100:.2f}% Calmar={metrics['calmar']:.3f} ({elapsed:.1f}s)")
    return {"config_name":cfg.name,"metrics":metrics,"dr_mss":dr_mss,"sub_drs":sub_drs,"elapsed":round(elapsed,1)}

# ═══════════════ 主流程 ═══════════════

def main():
    logger.info("="*70)
    logger.info("mss_dynamic 全量回测 + 归因分析")
    logger.info(f"数据: 最新至2026-06-02")
    logger.info("="*70)

    # 1. 加载数据
    data_tuple=load_data(start_date="2018-01-01", end_date="2026-06-02")
    data_ctx=DataContext(*data_tuple)
    nd=data_ctx.nd; ds=data_ctx.ds

    # 2. 运行完整回测：实盘口径 + 研究口径
    results={}
    for tpp,lbl in [(0.05,"实盘口径_t5"),(0.03,"实盘口径_t3")]:
        cfg=ExCfg(name=lbl,trail=tpp,top_n_mode="live")
        results[lbl]=run_experiment(data_ctx,cfg)
    for tpp,lbl in [(0.05,"研究口径_t5"),(0.03,"研究口径_t3")]:
        cfg=ExCfg(name=lbl,trail=tpp,top_n_mode="research")
        results[lbl]=run_experiment(data_ctx,cfg)

    # 3. 子策略独立回测 (实盘口径)
    signals=data_ctx.ensure_signals()
    symbol_risk_map=data_ctx.ensure_risk_map()
    states,confidence=data_ctx.ensure_states()
    sub_results={}
    for name,params in SUB_PARAMS.items():
        if name in ("mf_base","chip_rp"): continue
        dr=bt_sub_strategy(signals[params["signal"]],data_ctx.fwd,data_ctx.dm,
            rebal_freq=params["rf"],
            top_n=LIVE_TOP_N.get(name,params["tn"]),
            min_hold_days=params["mhd"],
            pos_ratio=signals.get("composite_p" if params.get("timing")=="composite" else params.get("timing")),
            stop_loss_pct=STOP_LOSS.get(name,0.06),
            symbol_risk_map=symbol_risk_map,
            use_trailing_stop=True,trailing_profit_pct=0.05,
            states=states)
        sub_results[name]=dr

    # 4. 窗口分析 (主策略)
    window_results={}
    for lbl in ["实盘口径_t5","实盘口径_t3","研究口径_t5","研究口径_t3"]:
        r=results[lbl]
        window_results[lbl]=window_analysis(r["dr_mss"],ds,lbl)

    # 5. 归因分析
    # 5a. 各市场状态分布
    state_counts={}
    for s in ["bull","bear","oscillate","recovery"]:
        state_counts[s]=sum(1 for st in states if st==s)
    logger.info(f"市场状态分布: {state_counts}")

    # 5b. 各子策略归因 (实盘口径t5)
    sub_drs=results["实盘口径_t5"]["sub_drs"]
    if "mf_base" in sub_drs and "chip_rp" in sub_drs:
        sub_drs["mf50_chip50"]=0.5*sub_drs["mf_base"]+0.5*sub_drs["chip_rp"]
    sds=[n for n in sub_drs if n in DEFAULT_ALLOC["bull"] or any(n in [a[0] for a in al] for al in DEFAULT_ALLOC.values())]
    sds=list(dict.fromkeys(sds))
    attr_state=attribution_by_state({n:sub_drs[n] for n in sds},states,nd)
    attr_contrib=attribution_sub_contribution(sub_drs,sds,states_series:=states,_to_state_alloc_format(DEFAULT_ALLOC),nd)
    corr=correlation_analysis({n:sub_drs[n] for n in sds if n in sub_drs})
    monthly=monthly_returns(results["实盘口径_t5"]["dr_mss"],ds,"mss")
    yearly=yearly_breakdown(results["实盘口径_t5"]["dr_mss"],ds)

    # 6. Walk-forward (3个窗口, 实盘口径)
    wf_results=[]
    windows_wf=[("2019-01~2022-01","2019-01-02","2022-01-01"),
                ("2020-01~2023-01","2020-01-02","2023-01-01"),
                ("2021-01~2024-01","2021-01-04","2024-01-01")]
    for wname,ts,te in windows_wf:
        logger.info(f"Walk-forward: {wname}")
        dt=load_data(start_date=ts,end_date=te)
        dc=DataContext(*dt)
        for tpp in [0.05,0.03]:
            cfg=ExCfg(name=f"wf_{wname}_t{int(tpp*100)}",trail=tpp,top_n_mode="live")
            r=run_experiment(dc,cfg); wf_results.append({"window":wname,"trail":f"{int(tpp*100)}%",**r})

    # 7. 保存结果
    output={
        "meta":{"data_end":"2026-06-02","n_days":nd,"n_stocks":data_ctx.ns},
        "main_strategy":{lbl:{"metrics":results[lbl]["metrics"],"windows":wr}
                         for lbl,wr in window_results.items()},
        "sub_strategies":{n:compute_metrics(sub_results[n],n) for n in sub_results},
        "state_attribution":attr_state,
        "contribution_attribution":{k:{kk:vv for kk,vv in v.items() if kk!="name"}
                                     for k,v in attr_contrib.items()},
        "correlation":corr,
        "monthly_returns":{str(k):v for k,v in monthly.items()},
        "yearly_returns":{str(k):v for k,v in yearly.items()},
        "walk_forward":wf_results,
        "state_counts":state_counts,
    }
    rp=os.path.join(RESULTS_DIR,"full_backtest.json")
    with open(rp,"w",encoding="utf-8") as f:
        json.dump(output,f,indent=2,ensure_ascii=False,default=str)
    logger.info(f"结果已保存至: {rp}")

    # 8. 打印摘要报告
    _print_report(results,window_results,sub_results,attr_state,attr_contrib,corr,monthly,yearly,wf_results,state_counts,nd)

def _print_report(results,wr,sub,attr_state,attr_contrib,corr,monthly,yearly,wf,state_counts,nd):
    print(); print("="*70)
    print(f"  📊 mss_dynamic 全量回测报告 ({nd}个交易日, 最新数据至2026-06-02)")
    print("="*70)

    print(); print("─"*70)
    print("  参数: 交易成本=0.12% | 增强ST过滤 | 市场广度降级 | V6a_3way分配")
    print("  实盘口径: mf top_n=10/8, chip top_n=6, osr top_n=6")
    print("  止损: mf=6%, chip=8% | 移动止盈: 5%/3%")
    print("─"*70)

    print(); print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  主策略全区间 (2019-01 ~ 2026-06, {nd}天)                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  {'口径':<12} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} ║")
    for lbl in ["实盘口径_t5","实盘口径_t3","研究口径_t5","研究口径_t3"]:
        m=results[lbl]["metrics"]
        sign="📈" if m["calmar"]>15 else "🟢"
        print(f"║ {sign} {lbl:<10} {m['annual_return']*100:>+7.2f} {m['sharpe']:>7.3f} {abs(m['max_drawdown'])*100:>7.2f} {m['calmar']:>7.3f} {m['win_rate']*100:>5.1f}% ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  分年窗口 (实盘口径 trail=5%)                                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  {'窗口':<10} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} ║")
    for w in wr["实盘口径_t5"]:
        if w.get("window","全区间")=="全区间": continue
        ar=w.get("annual_return",0); dd=abs(w.get("max_drawdown",0))
        ca=w.get("calmar",0); sp=w.get("sharpe",0); wr_=w.get("win_rate",0)
        print(f"║  {w.get('window',''):<10} {ar*100:>+7.2f} {sp:>7.3f} {dd*100:>7.2f} {ca:>7.3f} {wr_*100:>5.1f}% ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  子策略独立回测 (实盘口径 trail=5%)                             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  {'子策略':<14} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'角色':<12} ║")
    sub_roles={"mf_d10_rp":"主力多因子","mf_vol_d10_rp":"波动择时","chip_covrp":"防御核心",
               "chip_equal_d3":"等权防御","c01_layered_d5":"分层择时","osr_d10":"超跌反弹"}
    for n in ["mf_d10_rp","mf_vol_d10_rp","chip_covrp","chip_equal_d3","c01_layered_d5","osr_d10"]:
        s=sub.get(n,{})
        ar=s.get("annual_return",0); dd=abs(s.get("max_drawdown",0))
        ca=s.get("calmar",0); sp=s.get("sharpe",0)
        role=sub_roles.get(n,"")
        print(f"║  {n:<14} {ar*100:>+7.2f} {sp:>7.3f} {dd*100:>7.2f} {ca:>7.3f} {role:<12} ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  市场状态分布 (共{nd}天)                                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for st,cnt in sorted(state_counts.items()):
        pct=cnt/nd*100
        print(f"║  {st:<12}: {cnt:>5d}天 ({pct:>5.1f}%) ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  各市场状态下子策略日均收益贡献                                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for st,sds in attr_state.items():
        print(f"║  [{st}]")
        sorted_sds=sorted(sds.items(),key=lambda x: x[1]["mean_ret"],reverse=True)
        for n,data in sorted_sds:
            print(f"║    {n:<14} 日均={data['mean_ret']*100:>+.4f}% 胜率={data['win_rate']*100:>5.1f}% ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  子策略对主策略的加权贡献归因 (实盘口径t5)                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    contribs={k:v for k,v in attr_contrib.items() if k!="mss_total"}
    sorted_c=sorted(contribs.items(),key=lambda x: x[1]["annual_return"],reverse=True)
    for n,c in sorted_c:
        contrib_pct=c["annual_return"]/attr_contrib["mss_total"]["annual_return"]*100 if attr_contrib["mss_total"]["annual_return"]!=0 else 0
        print(f"║  {n:<14} 年化={c['annual_return']*100:>+6.2f}% 贡献={contrib_pct:>5.1f}% ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  子策略收益相关性                                            ║")
    corr_names=corr["names"]; corr_mat=corr["correlation"]
    header="".join(f"{n[:5]:>6}" for n in corr_names)
    print(f"║  {'':<14} {header} ║")
    for i,n in enumerate(corr_names):
        row="".join(f"{corr_mat[i][j]:>6.2f}" for j in range(len(corr_names)))
        print(f"║  {n:<14} {row} ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  逐年收益 (实盘口径t5)                                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for yr,ret in sorted(yearly.items()):
        print(f"║  {int(yr):>4}年: {ret*100:>+7.2f}% ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Walk-forward 跨窗口验证 (实盘口径)                              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  {'窗口':<18} {'trail':<6} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} ║")
    for w in wf:
        m=w["metrics"]
        print(f"║  {w['window']:<18} {w['trail']:<6} {m['annual_return']*100:>+7.2f} {m['sharpe']:>7.3f} {abs(m['max_drawdown'])*100:>7.2f} {m['calmar']:>7.3f} ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  月度收益矩阵 (实盘口径t5, 单位%)                               ║")
    months="1月 2月 3月 4月 5月 6月 7月 8月 9月 10月 11月 12月"
    print(f"║      {months} ║")
    for yr in sorted(monthly.keys()):
        row=f"{int(yr)}"
        for m in range(1,13):
            v=monthly[yr].get(m,0)
            if v>0: row+=f" {v*100:>+5.2f}"
            elif v<0: row+=f" {v*100:>+5.2f}"
            else: row+="   -  "
        print(f"║  {row} ║")

    print("╚══════════════════════════════════════════════════════════════╝")
    print(); print("报告完整保存至:", os.path.join(RESULTS_DIR,"full_backtest.json"))

if __name__=="__main__":
    main()
