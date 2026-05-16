"""
X1 — 多策略组合 + Chip窗口验证 + 费率测试 + 涨跌停过滤
"""
import os,sys,json,logging,gc,numpy as np,pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)]);logger=logging.getLogger("X1")
TX=0.0012

FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99','returns','rsi_14','volatility_20','macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position']))

WINDOWS=[
    ("2019修复牛","2019-01-02","2019-12-31"),
    ("2020疫情","2020-01-02","2020-12-31"),
    ("2021结构牛","2021-01-04","2021-12-31"),
    ("2022熊市","2022-01-04","2022-12-30"),
    ("2023震荡","2023-01-03","2023-12-29"),
    ("2024反弹","2024-01-02","2024-12-31"),
    ("2025至今","2025-01-02","2026-04-30"),
]

RESULTS_DIR=os.path.join(os.path.dirname(__file__),"x1_results");os.makedirs(RESULTS_DIR,exist_ok=True)

def load():
    db=Database()
    df=db.get_factors(start_date="2018-01-01",end_date="2026-04-30",factor_names=FACTORS,with_close=True)
    df['date']=pd.to_datetime(df['date']);ds=sorted(df['date'].unique())
    tks=db.get_symbols()['symbol'].tolist();nd,ns,nf=len(ds),len(tks),len(FACTORS)
    t2i={t:i for i,t in enumerate(tks)};d2i={d:i for i,d in enumerate(ds)}
    v3=np.full((nd,ns,nf),np.nan,dtype=np.float32);dm=np.zeros((nd,ns),dtype=bool);cl=np.zeros((nd,ns),dtype=np.float32)
    di=np.array([d2i[d] for d in df['date']],dtype=np.int32)
    si=np.array([t2i.get(s,-1) for s in df['symbol']],dtype=np.int32)
    v=si>=0;di,si=di[v],si[v]
    for fi,fc in enumerate(FACTORS):
        if fc in df.columns:v3[di,si,fi]=df[fc].values[v].astype(np.float32)
    cl[di,si]=df['close'].values[v].astype(np.float32);dm[di,si]=True
    np.nan_to_num(v3,nan=0.0,copy=False);np.nan_to_num(cl,nan=0.0,copy=False)
    fwd=np.zeros((nd,ns),dtype=np.float32)
    for d in range(nd-1):
        b=(cl[d]>1e-10)&(cl[d+1]>1e-10);fwd[d,b]=(cl[d+1,b]-cl[d,b])/cl[d,b]
    z3=np.zeros_like(v3)
    for fi in range(nf):
        a=v3[:,:,fi]
        for d in range(nd):
            r=a[d,:];nz=r[r!=0]
            if len(nz)>1:lo,hi=np.quantile(nz,[0.01,0.99]);c=np.clip(r,lo,hi);mu,sd=np.mean(c),np.std(c);z3[d,:,fi]=(c-mu)/sd if sd>1e-10 else 0.0
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子")
    return z3,fwd,dm,tks,FACTORS,nd,ns,ds

def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

def bt(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None,mhd=5,tx=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);ttx=0.0;nt=0
    for i in range(1,nd):
        rebal=(i%rf==0);txc=0.0
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*tx;ttx+=txc
            if to>0.01:nt+=1
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):p2=pw[mk].copy()/float(np.sum(pw[mk]));pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i]))
        rt=0.0 if(np.isnan(rt)or np.isinf(rt))else rt
        dr[i]=rt-txc;eq[i]=eq[i-1]*(1.0+rt-txc);rh+=1
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

def bt_series(sig,fwd,dm,rf=3,tn=40,pos_ratio=None,mhd=5,tx=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;dr=np.zeros(nd,dtype=np.float64)
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):p2=pw[mk].copy()/float(np.sum(pw[mk]));pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i]))
        dr[i]=0.0 if(np.isnan(rt)or np.isinf(rt))else rt
        rh+=1
    return dr

def bt_series_with_tx(sig,fwd,dm,rf=3,tn=40,pos_ratio=None,mhd=5,tx=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;dr=np.zeros(nd,dtype=np.float64)
    for i in range(1,nd):
        rebal=(i%rf==0);txc=0.0
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*tx
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):p2=pw[mk].copy()/float(np.sum(pw[mk]));pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i]))
        dr[i]=0.0 if(np.isnan(rt)or np.isinf(rt))else rt-txc
        rh+=1
    return dr

def combo_from_series(dr1,dr2,dr3=None,w1=0.5,w2=0.5,w3=0.0):
    nd=len(dr1);eq=np.ones(nd)
    dr=dr1*w1+dr2*w2+(dr3*w3 if dr3 is not None else 0)
    for i in range(1,nd):eq[i]=eq[i-1]*(1.0+dr[i])
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    return{"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":0}

def bt_window(sig,fwd,dm,ds,name,rf=3,tn=40,pos_ratio=None,mhd=5,tx=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);nt=0
    for i in range(1,nd):
        rebal=(i%rf==0);txc=0.0
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*tx
            if to>0.01:nt+=1
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):p2=pw[mk].copy()/float(np.sum(pw[mk]));pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i]))
        rt=0.0 if(np.isnan(rt)or np.isinf(rt))else rt
        dr[i]=rt-txc;eq[i]=eq[i-1]*(1.0+rt-txc);rh+=1
    total_metrics=calc_metrics(eq,dr,nd,name,"全区间",nt)
    win_results=[total_metrics]
    for wname,ws,we in WINDOWS:
        try:
            ws_d=pd.Timestamp(ws);we_d=pd.Timestamp(we)
            in_window=[j for j,d in enumerate(ds) if ws_d<=d<=we_d]
            if len(in_window)<5:
                win_results.append({"window":wname,"annual_return":0,"sharpe":0,"max_drawdown":0,"calmar":0,"win_rate":0,"n_trades":0,"n_days":0})
                continue
            w_start=in_window[0];w_end=in_window[-1]+1
            w_eq=eq[w_start:w_end];w_dr=dr[w_start:w_end];w_dr[0]=0.0
            wr=calc_metrics(w_eq,w_dr,len(w_eq),name,wname,nt)
            wr["n_days"]=len(in_window)
            win_results.append(wr)
        except Exception as e:
            win_results.append({"window":wname,"error":str(e)})
    return total_metrics,win_results

def calc_metrics(eq,dr,nd,name,wname,nt):
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}][{wname}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return{"name":name,"window":wname,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

def bt_limit_filter(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None,mhd=5,tx=TX):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);ttx=0.0;nt=0;nfilt=0
    for i in range(1,nd):
        rebal=(i%rf==0);txc=0.0
        if rebal:
            locked_idx=np.where(np.abs(fwd[i])>0.095)[0]
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            sig_i=sig[i].copy()
            sig_i[locked_idx]=-np.inf
            nfilt+=len(locked_idx)
            nw=alloc.allocate(sig_i,fwd,i,pw,hs,rh)
            to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*tx;ttx+=txc
            if to>0.01:nt+=1
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):p2=pw[mk].copy()/float(np.sum(pw[mk]));pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i]))
        rt=0.0 if(np.isnan(rt)or np.isinf(rt))else rt
        dr[i]=rt-txc;eq[i]=eq[i-1]*(1.0+rt-txc);rh+=1
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}+限价] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f} 过滤={nfilt}次")
    return{"name":name+"_限价过滤","annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt,"n_filters":nfilt}

def main():
    logger.info("="*60+"\nX1 多策略实验管道\n"+"="*60)
    z3,fwd,dm,tks,fnames,nd,ns,ds=load()
    fi={fn:i for i,fn in enumerate(fnames)}

    # ── MF 信号 (v1w参考权重) ──
    v1w_dict=v1w()
    wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # ── 择时信号 ──
    iv=fi.get('volatility_20')
    vol_p=np.ones(nd,dtype=np.float32)
    if iv is not None:vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.05,axis=1),0.2,1.0)

    # ── Chip 信号 (低波动+低动量) ──
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    vol20_idx=fi.get('volatility_20');m20_idx=fi.get('momentum_20')
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None:s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5
        if m20_idx is not None:s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3
        chip_sig[d]=np.nan_to_num(s,nan=-1e10)

    all_results={"data_info":{"nd":nd,"ns":ns,"nf":len(fnames)}}

    # ============================================================
    # A. 多策略组合（资金分配法）
    # ============================================================
    logger.info("\n"+ "="*60+"\nA. 多策略组合\n"+"="*60)

    logger.info("生成子策略收益序列...")

    # 子策略1: MF_D10_mhd=10
    dr_mf_d10=bt_series_with_tx(mf,fwd,dm,rf=10,tn=50,mhd=10,tx=TX)
    # 子策略2: Chip_D3
    dr_chip_d3=bt_series_with_tx(chip_sig,fwd,dm,rf=3,tn=40,mhd=5,tx=TX)
    # 子策略3: Chip_Vol_D3
    dr_chip_vol=bt_series_with_tx(chip_sig,fwd,dm,rf=3,tn=40,mhd=5,pos_ratio=vol_p,tx=TX)
    # 子策略4: MF+Vol_D10_mhd=10
    dr_mf_vol=bt_series_with_tx(mf,fwd,dm,rf=10,tn=50,mhd=10,pos_ratio=vol_p,tx=TX)

    # 单独评估各策略
    solo_names=["MF_D10_mhd10","Chip_D3","Chip_Vol_D3","MF_Vol_D10_mhd10"]
    solo_series=[dr_mf_d10,dr_chip_d3,dr_chip_vol,dr_mf_vol]
    solo_results=[]
    for sn,ss in zip(solo_names,solo_series):
        r=combo_from_series(ss,np.zeros(nd),w1=1.0,w2=0.0)
        r["name"]=sn
        solo_results.append(r)
        logger.info(f"  {sn}: Sharpe={r['sharpe']:.3f} 年化={r['annual_return']*100:.2f}%")

    combo_groups={}

    # A1. MF_D10 × Chip_D3
    logger.info("\n--- A1. MF_D10 × Chip_D3 ---")
    group1=[]
    for w1 in np.arange(0.1,1.0,0.1):
        w1r=round(w1,1);w2r=round(1.0-w1,1)
        r=combo_from_series(dr_mf_d10,dr_chip_d3,w1=w1r,w2=w2r)
        r["name"]=f"MF_D10({w1r})×Chip_D3({w2r})";r["w1"]=w1r;r["w2"]=w2r
        group1.append(r)
    combo_groups["MF_D10_x_Chip_D3"]=group1

    # A2. MF_D10 × Chip_Vol_D3
    logger.info("\n--- A2. MF_D10 × Chip_Vol_D3 ---")
    group2=[]
    for w1 in np.arange(0.1,1.0,0.1):
        w1r=round(w1,1);w2r=round(1.0-w1,1)
        r=combo_from_series(dr_mf_d10,dr_chip_vol,w1=w1r,w2=w2r)
        r["name"]=f"MF_D10({w1r})×Chip_Vol({w2r})";r["w1"]=w1r;r["w2"]=w2r
        group2.append(r)
    combo_groups["MF_D10_x_Chip_Vol"]=group2

    # A3. MF_Vol_D10 × Chip_D3
    logger.info("\n--- A3. MF_Vol_D10 × Chip_D3 ---")
    group3=[]
    for w1 in np.arange(0.1,1.0,0.1):
        w1r=round(w1,1);w2r=round(1.0-w1,1)
        r=combo_from_series(dr_mf_vol,dr_chip_d3,w1=w1r,w2=w2r)
        r["name"]=f"MF_Vol_D10({w1r})×Chip_D3({w2r})";r["w1"]=w1r;r["w2"]=w2r
        group3.append(r)
    combo_groups["MF_Vol_D10_x_Chip_D3"]=group3

    # A4. MF_Vol_D10 × Chip_D3 × Chip_Vol (三路)
    logger.info("\n--- A4. MF_Vol_D10 × Chip_D3 × Chip_Vol ---")
    triples=[
        ("等权(0.33/0.33/0.33)",0.33,0.33,0.34),
        ("偏MF(0.5/0.25/0.25)",0.5,0.25,0.25),
    ]
    group4=[]
    for tname,tw1,tw2,tw3 in triples:
        r=combo_from_series(dr_mf_vol,dr_chip_d3,dr_chip_vol,w1=tw1,w2=tw2,w3=tw3)
        r["name"]=f"MF_Vol×Chip×ChipVol_{tname}";r["w1"]=tw1;r["w2"]=tw2;r["w3"]=tw3
        group4.append(r)
    combo_groups["MF_Vol_D10_x_Chip_D3_x_Chip_Vol"]=group4

    # 合并所有组合，按Sharpe排序取前10
    all_combos=[]
    for grp in combo_groups.values():
        all_combos.extend(grp)
    all_combos_sorted=sorted(all_combos,key=lambda x:x["sharpe"],reverse=True)
    top10=all_combos_sorted[:10]

    logger.info("\n=== 多策略组合 Top10 (按Sharpe) ===")
    header=f"{'排名':<4} {'组合名称':<45} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}"
    logger.info(header)
    logger.info("-"*len(header))
    for idx,r in enumerate(top10,1):
        dd_str=f"{abs(r['max_drawdown'])*100:.1f}%" if r['max_drawdown']<0 else f"{r['max_drawdown']*100:.1f}%"
        logger.info(f"  {idx:<2} {r['name']:<45} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {dd_str:>7} {r['calmar']:>7.3f}")

    all_results["combo"]={
        "solo_strategies":solo_results,
        "groups":{k:{"description":k,"results":v} for k,v in combo_groups.items()},
        "top10_by_sharpe":top10,
    }

    # ============================================================
    # B. Chip窗口验证
    # ============================================================
    logger.info("\n"+ "="*60+"\nB. Chip窗口验证\n"+"="*60)

    chip_window_results=[]
    for sname,sig,rf_val,pr_val in [
        ("Chip_D3",chip_sig,3,None),
        ("Chip_Vol_D3",chip_sig,3,vol_p),
    ]:
        logger.info(f"\n窗口评估: {sname}")
        total,win_data=bt_window(sig,fwd,dm,ds,sname,rf=rf_val,tn=40,pos_ratio=pr_val,mhd=5,tx=TX)
        for w in win_data:
            w["strategy"]=sname
            chip_window_results.append(w)

    all_results["chip_window_validation"]=chip_window_results

    # 打印窗口验证表
    logger.info("\n=== Chip窗口验证结果 ===")
    hdr=f"{'策略':<16} {'窗口':<12} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6}"
    logger.info(hdr)
    logger.info("-"*len(hdr))
    for sname,_,_,_ in [("Chip_D3",chip_sig,3,None),("Chip_Vol_D3",chip_sig,3,vol_p)]:
        srows=[w for w in chip_window_results if w['strategy']==sname]
        for w in srows:
            dd_str=f"{abs(w['max_drawdown'])*100:.1f}%" if w['max_drawdown']<0 else f"{w['max_drawdown']*100:.1f}%"
            logger.info(f"  {sname:<14} {w['window']:<12} {w['annual_return']*100:>6.2f}% {w['sharpe']:>7.3f} {dd_str:>7} {w['calmar']:>7.3f} {w['win_rate']*100:>5.1f}%")

    # ============================================================
    # C. 费率对比
    # ============================================================
    logger.info("\n"+ "="*60+"\nC. 不同费率下最佳策略对比\n"+"="*60)

    best_strategies=[
        ("mf_vol_d10_mhd10",mf,10,vol_p,50,10),
        ("chip_d3",chip_sig,3,None,40,5),
        ("chip_vol_d3",chip_sig,3,vol_p,40,5),
        ("mf_d10",mf,10,None,50,10),
    ]

    fee_rates=[0.0012,0.0015,0.0020]
    fee_comparison=[]
    for sname,sig,rf_val,pr_val,tn,mhd in best_strategies:
        row={"strategy":sname}
        for tx_rate in fee_rates:
            r=bt(sig,fwd,dm,f"{sname}_tx{tx_rate}",rf=rf_val,tn=tn,pos_ratio=pr_val,mhd=mhd,tx=tx_rate)
            row[f"tx_{tx_rate}"]=r
        fee_comparison.append(row)
        logger.info(f"\n  {sname}:")
        for tx_rate in fee_rates:
            r=row[f"tx_{tx_rate}"]
            logger.info(f"    TX={tx_rate:.4f}: 年化={r['annual_return']*100:.2f}% Sharpe={r['sharpe']:.3f}")

    all_results["fee_comparison"]=fee_comparison

    # ============================================================
    # D. 涨跌停过滤
    # ============================================================
    logger.info("\n"+ "="*60+"\nD. 涨跌停过滤效果对比\n"+"="*60)

    limit_test_strategies=[
        ("mf_vol_d10_mhd10",mf,10,vol_p,50,10),
        ("chip_d3",chip_sig,3,None,40,5),
        ("chip_vol_d3",chip_sig,3,vol_p,40,5),
        ("mf_d10",mf,10,None,50,10),
    ]

    limit_comparison=[]
    for sname,sig,rf_val,pr_val,tn,mhd in limit_test_strategies:
        logger.info(f"\n  测试: {sname}")
        base=bt(sig,fwd,dm,sname,rf=rf_val,tn=tn,pos_ratio=pr_val,mhd=mhd,tx=TX)
        lim=bt_limit_filter(sig,fwd,dm,sname,rf=rf_val,tn=tn,pos_ratio=pr_val,mhd=mhd,tx=TX)
        limit_comparison.append({"strategy":sname,"baseline":base,"limit_filter":lim})
        logger.info(f"    基准: Sharpe={base['sharpe']:.3f} 年化={base['annual_return']*100:.2f}%")
        logger.info(f"    限价: Sharpe={lim['sharpe']:.3f} 年化={lim['annual_return']*100:.2f}%")

    all_results["limit_filter_comparison"]=limit_comparison

    # ── 保存 ──
    out_path=os.path.join(RESULTS_DIR,"results.json")
    with open(out_path,'w') as f:
        json.dump(all_results,f,indent=2,ensure_ascii=False,default=str)
    logger.info(f"\n✓ 结果已保存至: {out_path}")

    # ============================================================
    # 摘要打印
    # ============================================================
    logger.info("\n"+"="*80)
    logger.info("最终结果摘要")
    logger.info("="*80)

    logger.info("\n1. 多策略组合 Top10:")
    for idx,r in enumerate(top10,1):
        logger.info(f"   {idx:>2}. {r['name']:<50} Sharpe={r['sharpe']:.3f} 年化={r['annual_return']*100:.2f}%")

    logger.info("\n2. Chip窗口验证:")
    for sname,_,_,_ in [("Chip_D3",chip_sig,3,None),("Chip_Vol_D3",chip_sig,3,vol_p)]:
        srows=[w for w in chip_window_results if w['strategy']==sname and w['window']!="全区间"]
        pos=sum(1 for w in srows if w['annual_return']>0)
        logger.info(f"   {sname}: {pos}/{len(srows)} 窗口正收益")
        for w in srows:
            icon="✓" if w['annual_return']>0 else"✗"
            logger.info(f"     {icon} {w['window']:<10} {w['annual_return']*100:+.2f}%")

    logger.info("\n3. 费率对比 (最佳策略):")
    for row in fee_comparison:
        logger.info(f"   {row['strategy']}:")
        for tx_rate in fee_rates:
            r=row[f"tx_{tx_rate}"]
            logger.info(f"     TX={tx_rate:.4f} → Sharpe={r['sharpe']:.3f} 年化={r['annual_return']*100:.2f}%")

    logger.info("\n4. 涨跌停过滤效果:")
    for row in limit_comparison:
        b=row['baseline'];l=row['limit_filter']
        logger.info(f"   {row['strategy']}: 基准Sharpe={b['sharpe']:.3f} → 限价Sharpe={l['sharpe']:.3f}")

    logger.info(f"\n完整结果文件: {out_path}")
    return all_results,out_path

if __name__=="__main__":
    all_results,out_path=main()
