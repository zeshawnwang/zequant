"""V6 消融实验 — chip_v2信号源 + chip_covrp替换 + 2025→2026 Walk-forward

基于 comprehensive_backtest.py 框架，运行三个实验:
  1. chip_v2 信号源 vs 原始 (sig=v1 vs sig=v2) + trail=3%/5%
  2. chip_covrp 替换方案 (c01_layered_d5 / chip_vol_rp)
  3. Walk-forward 扩展: 2024-01→2026-06

用法: python3 daily/2026-06-02/experiments.py
"""
from __future__ import annotations
import copy, json, logging, os, sys, time
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb
from core.positioners import RPPortfolioWeights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(SCRIPT_DIR, "experiments.log"), mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger("mss_exp")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")
GA_CONFIG_PATH = os.path.abspath("./core/strategies/impl/v1_ga_rp/config.json")

FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99',
    'ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127',
    'gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168','gtja171','gtja176','gtja185','gtja34',
    'gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99','returns','rsi_14','volatility_20',
    'macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position','beta_20']))
NEW_FACTORS = ['ma5','ma10','ma20','ma21','ma60','ma120','ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross','volume_breakout_ratio','volume_contraction','chip_concentration','ma_angle_20']
ALL_FACTORS_V2 = list(set(FACTORS + NEW_FACTORS))

DEFAULT_ALLOC = {
    "bull":[("mf_d10_rp",0.6),("mf_vol_d10_rp",0.2),("chip_covrp",0.2)],
    "bear":[("chip_covrp",0.6),("chip_equal_d3",0.2),("mf_vol_d10_rp",0.2)],
    "oscillate":[("chip_covrp",0.4),("mf50_chip50",0.3),("c01_layered_d5",0.3)],
    "recovery":[("chip_covrp",0.4),("osr_d10",0.3),("mf_vol_d10_rp",0.3)]}

# 实验2: chip_covrp 替换方案
ALLOC_REPLACE_CHIP = {
    "bull":[("mf_d10_rp",0.6),("mf_vol_d10_rp",0.2),("c01_layered_d5",0.2)],
    "bear":[("c01_layered_d5",0.6),("chip_equal_d3",0.2),("mf_vol_d10_rp",0.2)],
    "oscillate":[("c01_layered_d5",0.4),("mf50_chip50",0.3),("c01_layered_d5",0.3)],
    "recovery":[("c01_layered_d5",0.4),("osr_d10",0.3),("mf_vol_d10_rp",0.3)]}

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

LIVE_TOP_N = {"mf_d10_rp":10,"mf_vol_d10_rp":8,"chip_covrp":6,"chip_equal_d3":6,
    "chip_rp":6,"osr_d10":6,"c01_layered_d5":6,"mf_base":6}

# 数据加载、信号构建、状态检测、回测引擎 — 与 comprehensive_backtest.py 相同
def _get_conn(): return duckdb.connect(DB_PATH, read_only=True)

def load_data(start_date="2018-01-01", end_date="2026-06-02", use_v2=False):
    t0=time.time(); conn=_get_conn()
    all_cols=[r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'").fetchall()]
    base=ALL_FACTORS_V2 if use_v2 else FACTORS
    available=[c for c in base if c in all_cols]
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
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子 ({time.time()-t0:.1f}s)")
    conn.close()
    return z3,fwd,dm,cl,tks,available,nd,ns,ds,t2i,per

def load_ga_weights():
    if os.path.exists(GA_CONFIG_PATH):
        with open(GA_CONFIG_PATH) as f: return json.load(f).get("selector",{}).get("weights",{})
    return {}

def build_signals(z3,fwd,dm,cl,fnames,nd,ns,ds,chip_v2=False):
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
    # chip_v2: 加入 chip_concentration / ma_alignment_score / volume_contraction
    if chip_v2:
        v2_idx={}; candidates=['chip_concentration','ma_alignment_score','volume_contraction']
        for c in candidates:
            if c in fi: v2_idx[c]=fi[c]
        if v2_idx:
            for d in range(nd):
                extra=np.zeros(ns)
                if 'chip_concentration' in v2_idx: extra+=np.where(z3[d,:,v2_idx['chip_concentration']]<-0.3,1.0,0.0)*0.3
                if 'ma_alignment_score' in v2_idx: extra+=np.where(z3[d,:,v2_idx['ma_alignment_score']]<0.0,1.0,0.0)*0.2
                if 'volume_contraction' in v2_idx: extra+=np.where(z3[d,:,v2_idx['volume_contraction']]<-0.3,1.0,0.0)*0.2
                chip_sig[d]=chip_sig[d]*0.7+np.nan_to_num(extra,nan=0.0)*0.3
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
        if m5_idx and m20_idx: m5v,m20v=z3[d,:,m5_idx],z3[d,:,m20_idx]; sl.append(np.where((m5v>0)&(m5v>m20v),1.0,np.where(m5v<0,0.0,0.5)))
        if ir: rv=z3[d,:,ir]; sl.append(np.where(rv>70,0.0,np.where(rv>=50,1.0,np.where(rv>=30,0.5,0.0))))
        if sl: trend_p[d]=np.clip(np.mean(np.mean(sl,axis=0)>=0.6)*2.0,0.1,1.0)
    composite_p=np.clip(trend_p*0.6+vol_p*0.4,0.1,1.0)
    mkt_idx=np.zeros(nd,dtype=np.float64)
    for d in range(1,nd):
        active=dm[d]&(cl[d]>1e-10)
        if np.any(active): mkt_idx[d]=np.mean(fwd[d-1,active])
    return {"mf":mf,"chip":chip_sig,"osr":osr_sig,"vol_p":vol_p,"trend_p":trend_p,"composite_p":composite_p,"fi":fi,"market_index":mkt_idx,"close":cl}

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

def run_mss(state_strategies,sub_drs,states,confidence,nd,breadth=None):
    dr=np.zeros(nd,dtype=np.float64)
    for i in range(1,nd):
        st=states[i] if i<len(states) else "oscillate"
        if breadth is not None and i<len(breadth) and breadth[i]<0.35: st="oscillate"
        allocs=state_strategies.get(st,state_strategies.get("oscillate",[]))
        am={}
        for a in allocs:
            if a["strategy"] in sub_drs: am[a["strategy"]]=max(a["weight"],0.0)
        tw=sum(am.values()) or 1
        for n in am: am[n]/=tw
        dr[i]=sum(am.get(n,0.0)*sub_drs[n][i] for n in am if n in sub_drs)
    return dr

def _to_state_alloc_format(alloc):
    return {st:[{"strategy":a[0],"weight":a[1]} for a in al] for st,al in alloc.items()}

# ═══════════════ 主实验流程 ═══════════════

@dataclass
class ExCfg:
    name:str=""; trail:float=0.05; use_v2:bool=False
    alloc:Optional[Dict]=None; top_n_mode:str="live"

class DataContext:
    def __init__(self,z3,fwd,dm,cl,tks,fnames,nd,ns,ds,t2i,per):
        self.z3,self.fwd,self.dm,self.cl=z3,fwd,dm,cl; self.tks,self.fnames=tks,fnames
        self.nd,self.ns,self.ds=nd,ns,ds; self.t2i,self.per=t2i,per
        self.signals,self.mkt_idx=None,None; self.symbol_risk_map,self.states,self.confidence=None,None,None
    def ensure_signals(self,chip_v2=False):
        if self.signals is None: self.signals=build_signals(self.z3,self.fwd,self.dm,self.cl,self.fnames,self.nd,self.ns,self.ds,chip_v2=chip_v2); self.mkt_idx=self.signals["market_index"]
        return self.signals
    def ensure_risk_map(self):
        if self.symbol_risk_map is None: self.symbol_risk_map=build_enhanced_st_mask(self.per,self.t2i,self.nd,self.ns)
        return self.symbol_risk_map
    def ensure_states(self):
        if self.states is None: self.ensure_signals(); self.states,self.confidence=detect_market_state(self.mkt_idx,self.nd)
        return self.states,self.confidence

def bt_sub_strategy(sig,fwd,dm,rf=10,top_n=50,mhd=10,pr=None,sl=0.06,risk_map=None,trail=0.05,states=None,tx=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=top_n,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32); hs=np.full(ns,-1,dtype=np.int32)
    rh,dr=0,np.zeros(nd,dtype=np.float64); ep=np.zeros(ns,dtype=np.float32); pp=np.zeros(ns)
    for i in range(1,nd):
        if sl>0:
            for j in range(ns):
                if pw[j]>0 and hs[j]>=0 and ep[j]>0 and fwd[i,j]<-sl and fwd[i,j]>-0.95:
                    pw[j]=0.0; hs[j]=-1; ep[j]=0.0; pp[j]=0.0
        if trail>0:
            for j in range(ns):
                if pw[j]>0 and ep[j]>0:
                    cur=ep[j]*(1.0+fwd[i,j])
                    if cur>pp[j] or pp[j]<=0: pp[j]=cur
                    if pp[j]>0 and cur<pp[j]*(1.0-trail): pw[j]=0.0; hs[j]=-1; ep[j]=0.0; pp[j]=0.0
        rebal=(i%rf==0)
        if rebal:
            masked=sig[i].copy()
            if risk_map:
                for j,lv in risk_map.items():
                    if lv=='high': masked[j]=-1e10
            nw=alloc.allocate(masked,fwd,i,pw,hs,rh)
            for j in range(ns):
                if nw[j]>0 and ep[j]<=0: ep[j]=max(1.0,1.0+fwd[i,j]); pp[j]=ep[j]
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
            if pw[j]>0 and ep[j]<=0: ep[j]=max(1.0,1.0+fwd[i,j])
        pr_val=pr[i] if pr is not None else 1.0
        rt=pr_val*float(np.dot(pw,fwd[i]))-0.5*to*tx
        dr[i]=0.0 if (np.isnan(rt) or np.isinf(rt)) else rt; rh+=1
    return dr

def run_experiment(data_ctx,cfg):
    t0=time.time(); logger.info(f"--- {cfg.name} ---")
    signals=data_ctx.ensure_signals(cfg.use_v2); nd=data_ctx.nd
    risk_map=data_ctx.ensure_risk_map(); states,confidence=data_ctx.ensure_states()
    breadth=compute_market_breadth(data_ctx.per["pct"],data_ctx.dm,nd)
    sp=copy.deepcopy(SUB_PARAMS)
    tn_map={"live":LIVE_TOP_N}
    tn=tn_map.get(cfg.top_n_mode,LIVE_TOP_N)
    for k in sp:
        if k in tn: sp[k]["tn"]=tn[k]
    sub_drs={}
    for name,params in sp.items():
        s=signals[params["signal"]]; pr=None; tm=params.get("timing")
        if tm=="vol": pr=signals["vol_p"]
        elif tm=="trend": pr=signals["trend_p"]
        elif tm=="composite": pr=signals["composite_p"]
        dr=bt_sub_strategy(s,data_ctx.fwd,data_ctx.dm,rf=params["rf"],
            top_n=params["tn"],mhd=params["mhd"],pr=pr,
            sl=STOP_LOSS.get(name,0.06),risk_map=risk_map,trail=cfg.trail,states=states)
        sub_drs[name]=dr
    if "mf_base" in sub_drs and "chip_rp" in sub_drs:
        sub_drs["mf50_chip50"]=0.5*sub_drs["mf_base"]+0.5*sub_drs["chip_rp"]
        sub_drs["mf60_chip40"]=0.6*sub_drs["mf_base"]+0.4*sub_drs["chip_rp"]
    alloc=cfg.alloc or DEFAULT_ALLOC
    ss=_to_state_alloc_format(alloc)
    dr_mss=run_mss(ss,sub_drs,states,confidence,nd,breadth=breadth)
    m=compute_metrics(dr_mss,cfg.name)
    logger.info(f"  {cfg.name}: AR={m['annual_return']*100:.2f}% SR={m['sharpe']:.3f} DD={abs(m['max_drawdown'])*100:.2f}% Calmar={m['calmar']:.3f} ({time.time()-t0:.1f}s)")
    return {"name":cfg.name,"metrics":m}

def main():
    logger.info("="*60)
    logger.info("V6 消融实验: chip_v2 + chip替换 + WF扩展")
    logger.info("="*60)

    # 加载基准数据 (FACTORS 不含NEW)
    dt=load_data("2018-01-01","2026-06-02",use_v2=False)
    dc=DataContext(*dt)

    results=[]

    # === 实验1: chip_v2 信号源 ===
    logger.info("\n"+"="*40)
    logger.info("实验1: chip_v2 信号源 (含 NEW_FACTORS)")
    logger.info("="*40)
    dt_v2=load_data("2018-01-01","2026-06-02",use_v2=True)
    dc_v2=DataContext(*dt_v2)
    for tpp,lbl in [(0.05,"chip_v2_t5"),(0.03,"chip_v2_t3")]:
        r=run_experiment(dc_v2,ExCfg(name=lbl,trail=tpp,use_v2=True)); results.append(r)

    # === 实验2: chip_covrp 替换为 c01_layered_d5 ===
    logger.info("\n"+"="*40)
    logger.info("实验2: chip_covrp→c01_layered_d5 替换")
    logger.info("="*40)
    for tpp,lbl in [(0.05,"chip_replace_c01_t5"),(0.03,"chip_replace_c01_t3")]:
        r=run_experiment(dc,ExCfg(name=lbl,trail=tpp,alloc=ALLOC_REPLACE_CHIP)); results.append(r)

    # === 实验3: Walk-forward 2024→2026 ===
    logger.info("\n"+"="*40)
    logger.info("实验3: Walk-forward 2024-01→2026-06")
    logger.info("="*40)
    dt_wf=load_data("2024-01-02","2026-06-02",use_v2=False)
    dc_wf=DataContext(*dt_wf)
    for tpp,lbl in [(0.05,"wf_2024_t5"),(0.03,"wf_2024_t3")]:
        r=run_experiment(dc_wf,ExCfg(name=lbl,trail=tpp)); results.append(r)

    # 保存结果
    rp=os.path.join(RESULTS_DIR,"experiments.json")
    with open(rp,"w",encoding="utf-8") as f:
        json.dump(results,f,indent=2,ensure_ascii=False,default=str)
    logger.info(f"结果已保存至: {rp}")

    # 打印对比表
    print("\n"+"="*60)
    print("📊 消融实验结果汇总")
    print("="*60)
    print(f"  {'实验':<25} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'对比':<8}")
    print("  "+"-"*65)
    refs={"实盘口径_t5":7.86,"实盘口径_t3":15.69}
    for r in results:
        m=r["metrics"]; name=r["name"]
        ref=refs.get("实盘口径_t5" if "t5" in name else "实盘口径_t3",0)
        delta=m["calmar"]-ref
        sign="+" if delta>0 else ""
        print(f"  {name:<25} {m['annual_return']*100:>+7.2f} {m['sharpe']:>7.3f} {abs(m['max_drawdown'])*100:>7.2f} {m['calmar']:>7.3f} {sign}{delta:>+.2f}")
    print("="*60)

if __name__=="__main__":
    main()
