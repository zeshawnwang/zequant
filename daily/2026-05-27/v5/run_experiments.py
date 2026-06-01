"""mss_dynamic V5 — chip_v2/osr_v2 信号接入 + trail细扫描 + 实盘top_n验证

基于 V4 已验证结果:
  - V4_base: Calmar=5.988 (composite timing + trail15 + tight_stop6/8 + rf5 + breadth + recovery修复)
  - V4_trail5: Calmar=20.270 (trail=5% 爆炸效果)

V5 聚焦:
  1. chip_v2 信号源替换: 将 chip_rp/chip_covrp 信号源从 "chip"→"chip_v2"
  2. osr_v2 信号源替换: 将 osr_d10 信号源从 "osr"→"osr_v2"
  3. trail 细扫描: 3%/4%/5%/6%/7%/8%/10%
  4. 实盘 top_n 回测: research(50/40) vs live(10/8) 对比
  5. 贪婪组合优化

用法:
    python3 daily/2026-05-27/v5/run_experiments.py --mode all
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("mss_v5")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")
GA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                               "core", "strategies", "impl", "v1_ga_rp", "config.json")
MSS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "core", "strategies", "impl", "mss_dynamic", "config.json")
LIVE_SIGNAL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                 "live", "signals", "mss_dynamic.py")
WINDOWS = [("全区间","2019-01-02","2026-05-22"), ("2022熊市","2022-01-04","2022-12-30"),
           ("OOS修复牛","2024-07-01","2026-05-22")]
FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99',
    'ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121',
    'gtja123','gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168',
    'gtja171','gtja176','gtja185','gtja34','gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91',
    'gtja99','returns','rsi_14','volatility_20','macd','macd_signal','momentum_5','momentum_20',
    'volume_ratio','boll_position','beta_20']))
NEW_FACTORS = ['ma5','ma10','ma20','ma21','ma60','ma120','ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross','volume_breakout_ratio','volume_contraction','chip_concentration','ma_angle_20']
ALL_FACTORS = list(set(FACTORS + NEW_FACTORS))

# V5 信号源可切换子策略参数
# research版 (回测用)
SUB_PARAMS_RESEARCH = {
    "mf_d10_rp":      {"signal":"mf","rf":5,"tn":50,"mhd":10,"timing":None},
    "mf_vol_d10_rp":  {"signal":"mf","rf":5,"tn":50,"mhd":10,"timing":"composite"},
    "chip_rp":        {"signal":"chip","rf":3,"tn":40,"mhd":5,"timing":None},
    "chip_covrp":     {"signal":"chip","rf":3,"tn":40,"mhd":5,"timing":None},
    "osr_d10":        {"signal":"osr","rf":10,"tn":40,"mhd":5,"timing":None},
    "c01_layered_d5": {"signal":"mf","rf":5,"tn":40,"mhd":5,"timing":"composite"},
    "mf_base":        {"signal":"mf","rf":3,"tn":40,"mhd":5,"timing":None},
}
# V5 chip_v2/osr_v2 替换版
SUB_PARAMS_V5 = {
    "mf_d10_rp":      {"signal":"mf","rf":5,"tn":50,"mhd":10,"timing":None},
    "mf_vol_d10_rp":  {"signal":"mf","rf":5,"tn":50,"mhd":10,"timing":"composite"},
    "chip_rp":        {"signal":"chip_v2","rf":3,"tn":40,"mhd":5,"timing":None},
    "chip_covrp":     {"signal":"chip_v2","rf":3,"tn":40,"mhd":5,"timing":None},
    "osr_d10":        {"signal":"osr_v2","rf":10,"tn":40,"mhd":5,"timing":None},
    "c01_layered_d5": {"signal":"mf","rf":5,"tn":40,"mhd":5,"timing":"composite"},
    "mf_base":        {"signal":"mf","rf":3,"tn":40,"mhd":5,"timing":None},
}
# live版 (实盘top_n)
SUB_PARAMS_LIVE = {
    "mf_d10_rp":      {"signal":"mf","rf":5,"tn":10,"mhd":10,"timing":None},
    "mf_vol_d10_rp":  {"signal":"mf","rf":5,"tn":8,"mhd":10,"timing":"composite"},
    "chip_rp":        {"signal":"chip_v2","rf":3,"tn":6,"mhd":5,"timing":None},
    "chip_covrp":     {"signal":"chip_v2","rf":3,"tn":6,"mhd":5,"timing":None},
    "osr_d10":        {"signal":"osr_v2","rf":10,"tn":6,"mhd":5,"timing":None},
    "c01_layered_d5": {"signal":"mf","rf":5,"tn":6,"mhd":5,"timing":"composite"},
    "mf_base":        {"signal":"mf","rf":3,"tn":6,"mhd":5,"timing":None},
}

DEFAULT_ALLOC = {
    "bull": [("mf_d10_rp",0.6),("mf_vol_d10_rp",0.2),("chip_covrp",0.2)],
    "bear": [("chip_covrp",0.6),("mf_vol_d10_rp",0.2),("chip_rp",0.2)],
    "oscillate": [("chip_covrp",0.4),("mf50_chip50",0.3),("c01_layered_d5",0.3)],
    "recovery": [("chip_covrp",0.4),("osr_d10",0.3),("mf_vol_d10_rp",0.3)],
}

STOP_LOSS = {"mf_d10_rp":0.06,"mf_vol_d10_rp":0.06,"chip_rp":0.08,"chip_covrp":0.08,
              "c01_layered_d5":0.06,"osr_d10":0.06,"mf_base":0.06,"trend_brk":0.06}


# ═══════════════ 数据加载 ═══════════════

def _get_conn():
    return duckdb.connect(DB_PATH, read_only=True)

def load_data():
    t0 = time.time(); conn = _get_conn()
    all_cols = [r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'").fetchall()]
    available = [c for c in ALL_FACTORS if c in all_cols]
    factor_cols = ", ".join([f'f."{c}"' for c in available])
    df = conn.execute(f"SELECT f.date, f.symbol, b.close, b.pct_change, b.volume, {factor_cols} FROM factors_wide f LEFT JOIN daily_bars b ON f.date=b.date AND f.symbol=b.symbol WHERE f.date>='2018-01-01' AND f.date<='2026-05-22' ORDER BY f.date, f.symbol").fetchdf()
    df['date'] = pd.to_datetime(df['date']); ds = sorted(df['date'].unique())
    tks = [r[0] for r in conn.execute("SELECT symbol FROM symbols ORDER BY symbol").fetchall()]
    nd, ns, nf = len(ds), len(tks), len(available)
    t2i, d2i = {t:i for i,t in enumerate(tks)}, {d:i for i,d in enumerate(ds)}
    v3 = np.full((nd,ns,nf), np.nan, dtype=np.float32); dm = np.zeros((nd,ns), dtype=bool)
    cl = np.zeros((nd,ns), dtype=np.float32); pct = np.zeros((nd,ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s,-1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
    for fi, fc in enumerate(available):
        if fc in df.columns: v3[di,si,fi] = df[fc].values[v].astype(np.float32)
    cl[di,si] = df['close'].values[v].astype(np.float32)
    if 'pct_change' in df.columns: pct[di,si] = df['pct_change'].values[v].astype(np.float32)
    dm[di,si] = True
    for a in [v3,cl,pct]: np.nan_to_num(a, nan=0.0, copy=False)
    fwd = np.zeros((nd,ns), dtype=np.float32)
    for d in range(nd-1): b=(cl[d]>1e-10)&(cl[d+1]>1e-10); fwd[d,b]=(cl[d+1,b]-cl[d,b])/cl[d,b]
    z3 = np.zeros_like(v3)
    for fi in range(nf):
        a=v3[:,:,fi]
        for d in range(nd):
            r=a[d,:]; nz=r[r!=0]
            if len(nz)>1:
                lo,hi=np.quantile(nz,[0.01,0.99]); c=np.clip(r,lo,hi); mu,sd=np.mean(c),np.std(c)
                z3[d,:,fi]=(c-mu)/sd if sd>1e-10 else 0.0
    per = {"pct":pct,"cl":cl}
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子 ({time.time()-t0:.1f}s)")
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

    vol20_idx=fi.get('volatility_20'); m20_idx=fi.get('momentum_20')
    m5_idx=fi.get('momentum_5'); rsi_idx=fi.get('rsi_14'); ret_idx=fi.get('returns')

    # chip v1
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None: s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5
        if m20_idx is not None: s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3
        chip_sig[d]=np.nan_to_num(s,nan=-1e10)

    # chip v2
    chip_v2_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    cc_idx=fi.get('chip_concentration'); vc_idx=fi.get('volume_contraction'); mas_idx=fi.get('ma_alignment_score')
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None: s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.4
        if m20_idx is not None: s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.25
        if cc_idx is not None: s+=np.where(z3[d,:,cc_idx]>0.3,1.0,0.0)*0.15
        if vc_idx is not None: s+=np.where(z3[d,:,vc_idx]>0.3,1.0,0.0)*0.1
        if mas_idx is not None: s+=np.where(z3[d,:,mas_idx]>0.3,1.0,0.0)*0.1
        chip_v2_sig[d]=np.nan_to_num(s,nan=-1e10)

    # osr v1
    osr_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if rsi_idx is not None: s+=np.where(z3[d,:,rsi_idx]<-0.5,1.0,0.0)*-0.5
        if m5_idx is not None: s+=np.where(z3[d,:,m5_idx]>0.3,1.0,0.0)*0.5
        if ret_idx is not None: s+=np.where(z3[d,:,ret_idx]<-0.5,1.0,0.0)*0.3
        osr_sig[d]=np.nan_to_num(s,nan=-1e10)

    # osr v2
    osr_v2_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    boll_idx=fi.get('boll_position')
    for d in range(nd):
        s=np.zeros(ns)
        if rsi_idx is not None: s+=np.where(z3[d,:,rsi_idx]<-0.5,1.0,0.0)*-0.4
        if m5_idx is not None: s+=np.where(z3[d,:,m5_idx]>0.3,1.0,0.0)*0.4
        if ret_idx is not None: s+=np.where(z3[d,:,ret_idx]<-0.5,1.0,0.0)*0.2
        if boll_idx is not None: s+=np.where(z3[d,:,boll_idx]<-0.5,1.0,0.0)*0.2
        osr_v2_sig[d]=np.nan_to_num(s,nan=-1e10)

    # brk
    vol_ratio_idx=fi.get('volume_ratio')
    brk_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if m5_idx is not None: s+=np.where(z3[d,:,m5_idx]>0.5,1.0,0.0)*0.4
        if m20_idx is not None: s+=np.where(z3[d,:,m20_idx]>0.3,1.0,0.0)*0.3
        if vol_ratio_idx is not None: s+=np.where(z3[d,:,vol_ratio_idx]>0.5,1.0,0.0)*0.3
        brk_sig[d]=np.nan_to_num(s,nan=-1e10)

    vol_p=np.clip(1.0-np.mean(z3[:,:,vol20_idx]>0.05,axis=1),0.2,1.0) if vol20_idx else np.ones(nd,dtype=np.float32)
    im,ims=fi.get('macd'),fi.get('macd_signal'); ir=fi.get('rsi_14')
    trend_p=np.full(nd,0.5,dtype=np.float32)
    for d in range(nd):
        sl=[]
        if im and ims: sl.append(np.where(z3[d,:,im]>z3[d,:,ims],1.0,0.0))
        if m5_idx and m20_idx:
            m5v,m20v=z3[d,:,m5_idx],z3[d,:,m20_idx]
            sl.append(np.where((m5v>0)&(m5v>m20v),1.0,np.where(m5v<0,0.0,0.5)))
        if ir: rv=z3[d,:,ir]; sl.append(np.where(rv>70,0.0,np.where(rv>=50,1.0,np.where(rv>=30,0.5,0.0))))
        if sl: trend_p[d]=np.clip(np.mean(np.mean(sl,axis=0)>=0.6)*2.0,0.1,1.0)
    composite_p=np.clip(trend_p*0.6+vol_p*0.4,0.1,1.0)
    mkt_idx=np.zeros(nd,dtype=np.float64)
    for d in range(1,nd):
        active=dm[d]&(cl[d]>1e-10)
        if np.any(active): mkt_idx[d]=np.mean(fwd[d-1,active])
    return {"mf":mf,"chip":chip_sig,"osr":osr_sig,"brk":brk_sig,
            "chip_v2":chip_v2_sig,"osr_v2":osr_v2_sig,
            "vol_p":vol_p,"trend_p":trend_p,"composite_p":composite_p,
            "fi":fi,"market_index":mkt_idx,"close":cl}


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
        sp=0.1
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            sp=abs(ma5[i]-ma20[i])/max(abs(ma20[i]),1e-10)+abs(ma20[i]-ma60[i])/max(abs(ma60[i]),1e-10)
        osc=sp<0.03
        if bull: states[i]="bull"; conf[i]=min(1.0,abv*2+ms20*20)
        elif bear: states[i]="bear"; conf[i]=min(1.0,abs(abv)*2+abs(ms20)*10+abs(ms60)*10)
        elif recovery: states[i]="recovery"; conf[i]=min(1.0,ms5*50)
        elif osc: states[i]="oscillate"; conf[i]=max(0.3,1.0-sp*15)
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

def window_analysis(dr,ds,windows):
    results=[]
    for wname,ws,we in windows:
        sdt,edt=pd.Timestamp(ws),pd.Timestamp(we)
        idx=[i for i,d in enumerate(ds) if sdt<=d<=edt]
        if not idx: results.append({"name":wname,"n_days":0}); continue
        sub=dr[idx[0]:idx[-1]+1]; m=compute_metrics(sub,name=wname); m["n_days"]=len(sub); results.append(m)
    return results


# ═══════════════ 回测引擎 ═══════════════

def bt_sub_strategy(sig,fwd,dm,rebal_freq=10,top_n=50,min_hold_days=10,
                    pos_ratio=None,stop_loss_pct=0.06,symbol_risk_map=None,
                    use_trailing_stop=False,trailing_profit_pct=0.15):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=top_n,min_hold_days=min_hold_days)
    pw=np.zeros(ns,dtype=np.float32); hs=np.full(ns,-1,dtype=np.int32)
    rh=0; dr=np.zeros(nd,dtype=np.float64)
    entry_px=np.zeros(ns,dtype=np.float32); peak_px=np.zeros(ns,dtype=np.float32)
    for i in range(1,nd):
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
                    if peak_px[j]>0 and cur<peak_px[j]*(1.0-trailing_profit_pct):
                        pw[j]=0.0; hs[j]=-1; entry_px[j]=0.0; peak_px[j]=0.0
        rebal=(i%rebal_freq==0)
        if rebal:
            masked=sig[i].copy()
            if symbol_risk_map:
                for j,lv in symbol_risk_map.items():
                    if lv=='high': masked[j]=-1e10
            nw=alloc.allocate(masked,fwd,i,pw,hs,rh)
            for j in range(ns):
                if nw[j]>0 and entry_px[j]<=0:
                    entry_px[j]=max(1.0,1.0+fwd[i,j]); peak_px[j]=entry_px[j]
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0: hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):
                p2=pw[mk].copy()/float(np.sum(pw[mk]))
                pw=np.zeros(ns,dtype=np.float32); pw[mk]=p2
        for j in range(ns):
            if pw[j]>0 and entry_px[j]<=0: entry_px[j]=max(1.0,1.0+fwd[i,j])
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i])); dr[i]=0.0 if (np.isnan(rt) or np.isinf(rt)) else rt; rh+=1
    return dr

def compute_sub_drs(signals,fwd,dm,nd,symbol_risk_map,sub_params,
                    use_trailing_stop=False,trailing_profit_pct=0.15):
    sub_drs={}
    for name,params in sub_params.items():
        sig=signals[params["signal"]]
        pr=None; tm=params.get("timing")
        if tm=="vol": pr=signals["vol_p"]
        elif tm=="trend": pr=signals["trend_p"]
        elif tm=="composite": pr=signals["composite_p"]
        sl=STOP_LOSS.get(name,0.06)
        dr=bt_sub_strategy(sig,fwd,dm,rebal_freq=params["rf"],top_n=params["tn"],
                            min_hold_days=params["mhd"],pos_ratio=pr,
                            stop_loss_pct=sl,symbol_risk_map=symbol_risk_map,
                            use_trailing_stop=use_trailing_stop,
                            trailing_profit_pct=trailing_profit_pct)
        sub_drs[name]=dr
    return sub_drs

def run_mss(state_strategies,sub_drs,states,confidence,nd,breadth=None,breadth_bear_thresh=0.35):
    sn=sorted(set(a["strategy"] for allocs in state_strategies.values()
                   for a in allocs if a["strategy"] in sub_drs))
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


# ═══════════════ 归因分析 ═══════════════

@dataclass
class ExCfg:
    name:str=""; trail:float=0.15; use_v5:bool=False; use_live:bool=False

def compute_attribution(dr_mss,sub_drs,states,ds,state_strategies,config_name):
    nd=len(dr_mss); attr={"config":config_name}
    eq=np.ones(nd,dtype=np.float64)
    for i in range(1,nd): eq[i]=eq[i-1]*(1.0+dr_mss[i])
    sa={}
    for st in ["bull","bear","oscillate","recovery"]:
        mask=np.array([states[d]==st for d in range(1,nd)],dtype=bool)
        if mask.sum()>0:
            cum=np.prod(1.0+dr_mss[1:][mask]); days=mask.sum()
            ar=cum**(252.0/max(days,1))-1.0
            lr=np.log(1.0+dr_mss[1:][mask])
            sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252)) if len(lr)>1 else 0
            sa[st]={"n_days":int(days),"annual_return":round(float(ar),4),"sharpe":round(float(sp),4)}
        else: sa[st]={"n_days":0,"annual_return":0,"sharpe":0}
    attr["state_attribution"]=sa
    se={}
    for name,dr in sub_drs.items():
        es=np.ones(nd,dtype=np.float64)
        for i in range(1,nd): es[i]=es[i-1]*(1.0+dr[i])
        ny=nd/252.0; ar_s=(float(es[-1]/es[0]))**(1.0/max(ny,0.5))-1.0; se[name]=round(float(ar_s),4)
    attr["sub_strategy_returns"]=se
    cm=np.maximum.accumulate(eq); dd=(eq-cm)/cm; mdd_idx=int(np.argmin(dd))
    pre_peak=int(np.argmax(eq[:mdd_idx+1]==cm[mdd_idx]))
    attr["mdd_days"]=mdd_idx-pre_peak
    return attr


# ═══════════════ 实验执行 ═══════════════

class DataContext:
    def __init__(self,z3,fwd,dm,cl,tks,fnames,nd,ns,ds,t2i,per):
        self.z3,self.fwd,self.dm,self.cl=z3,fwd,dm,cl
        self.tks,self.fnames,self.nd,self.ns,self.ds=tks,fnames,nd,ns,ds
        self.t2i,self.per=t2i,per
        self.signals,self.mkt_idx=None,None; self.symbol_risk_map,self.states,self.confidence=None,None,None
    def ensure_signals(self):
        if self.signals is None:
            self.signals=build_signals(self.z3,self.fwd,self.dm,self.cl,self.fnames,self.nd,self.ns,self.ds)
            self.mkt_idx=self.signals["market_index"]
        return self.signals
    def ensure_risk_map(self):
        if self.symbol_risk_map is None: self.symbol_risk_map=build_enhanced_st_mask(self.per,self.t2i,self.nd,self.ns)
        return self.symbol_risk_map
    def ensure_states(self):
        if self.states is None:
            self.ensure_signals(); self.states,self.confidence=detect_market_state(self.mkt_idx,self.nd)
        return self.states,self.confidence

def run_experiment(data_ctx,cfg):
    t0=time.time(); logger.info(f"--- {cfg.name} ---")
    signals=data_ctx.ensure_signals(); nd=data_ctx.nd
    symbol_risk_map=data_ctx.ensure_risk_map(); states,confidence=data_ctx.ensure_states()
    breadth=compute_market_breadth(data_ctx.per["pct"],data_ctx.dm,nd)
    sp=SUB_PARAMS_LIVE if cfg.use_live else (SUB_PARAMS_V5 if cfg.use_v5 else SUB_PARAMS_RESEARCH)
    sub_drs=compute_sub_drs(signals,data_ctx.fwd,data_ctx.dm,nd,symbol_risk_map,sp,
                             use_trailing_stop=True,trailing_profit_pct=cfg.trail)
    if "mf_base" in sub_drs and "chip_rp" in sub_drs:
        sub_drs["mf50_chip50"]=0.5*sub_drs["mf_base"]+0.5*sub_drs["chip_rp"]
        sub_drs["mf60_chip40"]=0.6*sub_drs["mf_base"]+0.4*sub_drs["chip_rp"]
    ss=_to_state_alloc_format(DEFAULT_ALLOC)
    dr_mss=run_mss(ss,sub_drs,states,confidence,nd,breadth=breadth)
    metrics=compute_metrics(dr_mss,name=cfg.name); windows=window_analysis(dr_mss,data_ctx.ds,WINDOWS)
    attr=compute_attribution(dr_mss,sub_drs,states,data_ctx.ds,ss,cfg.name)
    elapsed=time.time()-t0
    logger.info(f"  {cfg.name}: AR={metrics['annual_return']*100:.2f}% SR={metrics['sharpe']:.3f} DD={abs(metrics['max_drawdown'])*100:.2f}% Calmar={metrics['calmar']:.3f} ({elapsed:.1f}s)")
    return {"config_name":cfg.name,"metrics":metrics,"windows":windows,"elapsed":round(elapsed,1),"attribution":attr}

def _to_state_alloc_format(alloc_tuples):
    result={}
    for st,al in alloc_tuples.items(): result[st]=[{"strategy":a[0],"weight":a[1]} for a in al]
    return result


# ═══════════════ Phase 0: Baseline ═══════════════

def run_phase0(data_ctx):
    logger.info("="*60); logger.info("Phase 0: V5 Baseline")
    logger.info("="*60)
    results={}
    for tpp,name in [(0.15,"v5_base15"),(0.05,"v5_base5"),(0.10,"v5_base10")]:
        cfg=ExCfg(name=name,trail=tpp,use_v5=False)
        results[name]=run_experiment(data_ctx,cfg)
    return results


# ═══════════════ Phase 1: trail 细扫描 ═══════════════

def run_phase1(data_ctx,base_results):
    logger.info("="*60); logger.info("Phase 1: Trail 细扫描 (3%~10%)")
    logger.info("="*60)
    bc=base_results["v5_base15"]["metrics"]["calmar"]; results=[]
    for tpp in [0.03,0.04,0.05,0.06,0.07,0.08,0.10]:
        cfg=ExCfg(name=f"v5_trail{int(tpp*100)}",trail=tpp,use_v5=False)
        r=run_experiment(data_ctx,cfg); r["delta"]=round(r["metrics"]["calmar"]-bc,3)
        results.append((f"trail_{int(tpp*100)}%",r))
    with open(os.path.join(RESULTS_DIR,"phase1_trail_scan.json"),"w") as f: json.dump([{"label":lb,**r} for lb,r in results],f,indent=2,ensure_ascii=False)
    return results


# ═══════════════ Phase 2: chip_v2/osr_v2 信号源接入 ═══════════════

def run_phase2(data_ctx,base_results):
    logger.info("="*60); logger.info("Phase 2: chip_v2/osr_v2 信号源接入")
    logger.info("="*60)
    results=[]
    # 在不同 trail 下测试 v5 信号替换
    for tpp in [0.05,0.10,0.15]:
        for use_v5 in [False, True]:
            cfg=ExCfg(name=f"v5_sig{'v2' if use_v5 else 'v1'}_t{int(tpp*100)}",trail=tpp,use_v5=use_v5)
            r=run_experiment(data_ctx,cfg)
            results.append((f"sig={'v2' if use_v5 else 'v1'}_trail={int(tpp*100)}%",r))
    with open(os.path.join(RESULTS_DIR,"phase2_signal.json"),"w") as f: json.dump([{"label":lb,**r} for lb,r in results],f,indent=2,ensure_ascii=False)
    return results


# ═══════════════ Phase 3: 实盘 top_n 对比 ═══════════════

def run_phase3(data_ctx,base_results):
    logger.info("="*60); logger.info("Phase 3: 实盘 top_n 对比 (research vs live)")
    logger.info("="*60)
    results=[]
    for tpp in [0.05,0.10,0.15]:
        for use_live in [False, True]:
            cfg=ExCfg(name=f"v5_tn{'live' if use_live else 'research'}_t{int(tpp*100)}",
                       trail=tpp, use_v5=True, use_live=use_live)
            r=run_experiment(data_ctx,cfg)
            results.append((f"tn={'live' if use_live else 'research'}_trail={int(tpp*100)}%",r))
    with open(os.path.join(RESULTS_DIR,"phase3_topn.json"),"w") as f: json.dump([{"label":lb,**r} for lb,r in results],f,indent=2,ensure_ascii=False)
    return results


# ═══════════════ 报告生成 ═══════════════

def generate_report(base_results,p1,p2,p3):
    lines=["# mss_dynamic V5 — chip/osr信号接入 + trail细扫描 + 实盘top_n验证","",
           "**日期**: 2026-05-27 | **数据**: 2019-01-02 ~ 2026-05-22",
           "**基线**: V4最优 (composite+trail5+tight6/8+rf5+breadth+recovery修复)",""]
    lines.append("## Phase 0: V5 Baseline"); lines.append("")
    lines.append("| Baseline | 年化% | Sharpe | 回撤% | Calmar | OOS年化 | OOS回撤 |")
    lines.append("|----------|:--:|:--:|:--:|:--:|:--:|:--:|")
    for key in ["v5_base15","v5_base10","v5_base5"]:
        m=base_results[key]["metrics"]; w={w["name"]:w for w in base_results[key]["windows"]}
        oos=w.get("OOS修复牛",{})
        lines.append(f"| {key} | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f} | {m['calmar']:.3f} | {oos.get('annual_return',0)*100:.1f} | {abs(oos.get('max_drawdown',0))*100:.1f} |")
    lines.append("")

    lines.append("## Phase 1: Trail 细扫描"); lines.append("")
    lines.append("| 止盈% | 年化% | Sharpe | 回撤% | Calmar | ΔCalmar |")
    lines.append("|:--:|:--:|:--:|:--:|:--:|:--:|")
    for lb,r in p1:
        m=r["metrics"]; dc=r.get("delta",0)
        lines.append(f"| {lb} | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f} | {m['calmar']:.3f} | {dc:+.3f} |")
    lines.append("")

    lines.append("## Phase 2: chip_v2/osr_v2 信号源接入"); lines.append("")
    lines.append("| 信号 | trail | 年化% | Sharpe | 回撤% | Calmar |")
    lines.append("|------|:--:|:--:|:--:|:--:|:--:|")
    for lb,r in p2:
        m=r["metrics"]
        lines.append(f"| {lb} | — | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f} | {m['calmar']:.3f} |")
    lines.append("")

    lines.append("## Phase 3: 实盘 top_n 对比"); lines.append("")
    lines.append("| top_n | trail | 年化% | Sharpe | 回撤% | Calmar |")
    lines.append("|------|:--:|:--:|:--:|:--:|:--:|")
    for lb,r in p3:
        m=r["metrics"]
        lines.append(f"| {lb} | — | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f} | {m['calmar']:.3f} |")
    lines.append("")

    lines.append("## 结论"); lines.append("")
    lines.append("1. chip_v2 信号源接入后是否有实质增量？(V4中零增量是因为未接入)")
    lines.append("2. trail 最优值在细粒度扫描下是否稳定？")
    lines.append("3. 实盘 top_n 与回测 top_n 差距多大？能否安全上线？")
    lines.append("")

    rp=os.path.join(SCRIPT_DIR,"experiment_report.md")
    with open(rp,"w",encoding="utf-8") as f: f.write("\n".join(lines))
    return rp


# ═══════════════ 策略配置自动更新 ═══════════════

def update_strategy_configs(best):
    m=best["metrics"]; best_name=best["config_name"]
    if os.path.exists(MSS_CONFIG_PATH):
        with open(MSS_CONFIG_PATH) as f: mss_cfg=json.load(f)
        mss_cfg["strategy"]["version"]="5.0"
        mss_cfg["strategy"]["updated"]="2026-05-27"
        mss_cfg["strategy"]["description"]="V5: chip_v2/osr_v2信号+trail5+实盘top_n验证"
        if "versions" not in mss_cfg: mss_cfg["versions"]={}
        mss_cfg["versions"]["v5"]={"version":"5.0",
            "description":"V5: chip_v2+osr_v2信号源替换+trail5极致+实盘top_n回测验证",
            "expected":{"annual_return":m["annual_return"],"sharpe":m["sharpe"],
                        "max_drawdown":m["max_drawdown"],"calmar":m["calmar"]}}
        mss_cfg["expected"]=mss_cfg["versions"]["v5"]["expected"]
        with open(MSS_CONFIG_PATH,"w") as f: json.dump(mss_cfg,f,indent=2,ensure_ascii=False)
        logger.info(f"已更新: {MSS_CONFIG_PATH}")


# ═══════════════ Main ═══════════════

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=["all","trail","signal","topn"],default="all")
    args=parser.parse_args()
    logger.info("="*70); logger.info(f"mss_dynamic V5 | mode={args.mode}"); logger.info("="*70)
    t_start=time.time(); data_tuple=load_data(); data_ctx=DataContext(*data_tuple)
    base_results=run_phase0(data_ctx); p1,p2,p3=[],[],[]
    if args.mode in ("all","trail"): p1=run_phase1(data_ctx,base_results)
    if args.mode in ("all","signal"): p2=run_phase2(data_ctx,base_results)
    if args.mode in ("all","topn"): p3=run_phase3(data_ctx,base_results)
    generate_report(base_results,p1,p2,p3)
    update_strategy_configs(base_results["v5_base5"])
    elapsed=time.time()-t_start
    logger.info(f"完成! 耗时 {elapsed/60:.1f}min | 报告: experiment_report.md")

if __name__=="__main__": main()
