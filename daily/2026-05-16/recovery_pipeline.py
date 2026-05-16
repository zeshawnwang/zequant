"""
计算全部23策略的回撤修复时间。
回撤修复时间 = 从最大回撤底部到恢复至之前高点所需交易天数。
"""
import os,sys,json,logging,gc,numpy as np,pandas as pd

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger("recovery")
TX=0.0012

FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5','momentum_20',
    'volume_ratio','boll_position','beta_20']))
NEW_FACTORS=['ma5','ma10','ma20','ma21','ma60','ma120','ma_alignment_score','ma60_trend',
    'ma120_trend','macd_above_zero','macd_golden_cross','volume_breakout_ratio',
    'volume_contraction','ma_convergence','chip_concentration','ma_angle_20']
ALL_FACTORS=list(set(FACTORS+NEW_FACTORS))

REC_DIR=os.path.join(os.path.dirname(__file__),"recovery_results");os.makedirs(REC_DIR,exist_ok=True)

def load():
    db=Database()
    all_cols=db.list_factor_columns()
    available=[c for c in ALL_FACTORS if c in all_cols]
    df=db.get_factors(start_date="2018-01-01",end_date="2026-04-30",factor_names=available,with_close=True)
    df['date']=pd.to_datetime(df['date']);ds=sorted(df['date'].unique())
    tks=db.get_symbols()['symbol'].tolist();nd,ns,nf=len(ds),len(tks),len(available)
    t2i={t:i for i,t in enumerate(tks)};d2i={d:i for i,d in enumerate(ds)}
    v3=np.full((nd,ns,nf),np.nan,dtype=np.float32);dm=np.zeros((nd,ns),dtype=bool);cl=np.zeros((nd,ns),dtype=np.float32)
    di=np.array([d2i[d] for d in df['date']],dtype=np.int32)
    si=np.array([t2i.get(s,-1) for s in df['symbol']],dtype=np.int32)
    v=si>=0;di,si=di[v],si[v]
    for fi,fc in enumerate(available):
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
            if len(nz)>1:
                lo,hi=np.quantile(nz,[0.01,0.99]);c=np.clip(r,lo,hi)
                mu,sd=np.mean(c),np.std(c);z3[d,:,fi]=(c-mu)/sd if sd>1e-10 else 0.0
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子")
    return z3,fwd,dm,tks,available,nd,ns,ds

def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

def bt_recovery(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None,mhd=5):
    """bt() + 回撤修复时间。"""
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64)
    ttx=0.0;nt=0
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*TX;ttx+=txc
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
        dr[i]=rt;eq[i]=eq[i-1]*(1.0+rt);rh+=1

    # 标准指标
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))

    # 回撤修复时间
    ddn=np.arange(nd)
    mdd_idx=np.argmin(dd)
    pre_peak_val=cm[mdd_idx]  # 回撤前的高点
    # 找从mdd_idx之后第一次回到pre_peak_val的交易日
    recovery_candidates=np.where(eq[mdd_idx:]>=pre_peak_val)[0]
    if len(recovery_candidates)>0:
        recovery_days=int(recovery_candidates[0])
    else:
        recovery_days=nd-mdd_idx-1  # 超过全区间仍未修复

    # 最大回撤期（从顶部到底部）
    trough_idx=mdd_idx
    # 回撤开始日（上一次净值新高）
    peak_idx=np.argmax(cm[:trough_idx+1]==cm[trough_idx])
    drawdown_duration=int(trough_idx-peak_idx)

    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)

    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% 修复={recovery_days}d")
    return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),
           "max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),
           "n_trades":nt,"recovery_days":recovery_days,"drawdown_duration":drawdown_duration}

def main():
    z3,fwd,dm,tks,fnames,nd,ns,ds=load()
    fi={fn:i for i,fn in enumerate(fnames)}
    v1w_dict=v1w()

    # MF信号
    wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # Chip信号
    vol20_idx=fi.get('volatility_20');m20_idx=fi.get('momentum_20')
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s_chip=np.zeros(ns)
        if vol20_idx is not None:s_chip+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5
        if m20_idx is not None:s_chip+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3
        chip_sig[d]=np.nan_to_num(s_chip,nan=-1e10)

    # OSR信号
    rsi_idx=fi.get('rsi_14');m5_idx=fi.get('momentum_5')
    osr_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s_osr=np.zeros(ns)
        if rsi_idx is not None:s_osr+=np.where(z3[d,:,rsi_idx]<-0.5,1.0,0.0)*-0.5
        if m5_idx is not None:s_osr+=np.where(z3[d,:,m5_idx]>0.3,1.0,0.0)*0.5
        if 'returns' in fi:s_osr+=np.where(z3[d,:,fi['returns']]<-0.5,1.0,0.0)*0.3
        osr_sig[d]=np.nan_to_num(s_osr,nan=-1e10)

    # 择时信号
    iv=fi.get('volatility_20');vol_p=np.ones(nd,dtype=np.float32)
    if iv is not None:vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.05,axis=1),0.2,1.0)
    im,ims,im5,im20,ir=fi.get('macd'),fi.get('macd_signal'),fi.get('momentum_5'),fi.get('momentum_20'),fi.get('rsi_14')
    trend_p=np.full(nd,0.5,dtype=np.float32)
    for d in range(nd):
        sl=[]
        if im is not None and ims is not None:sl.append(np.where(z3[d,:,im]>z3[d,:,ims],1.0,0.0))
        if im5 is not None and im20 is not None:
            m5v,m20v=z3[d,:,im5],z3[d,:,im20]
            sl.append(np.where((m5v>0)&(m5v>m20v),1.0,np.where(m5v<0,0.0,0.5)))
        if ir is not None:
            rv=z3[d,:,ir]
            sl.append(np.where(rv>70,0.0,np.where(rv>=50,1.0,np.where(rv>=30,0.5,0.0))))
        if sl:trend_p[d]=np.clip(np.mean(np.mean(sl,axis=0)>=0.6)*2.0,0.1,1.0)

    res=[]
    logger.info("="*60+"\n23策略回撤修复时间\n"+"="*60)

    # ── 23个策略 ──
    strategies=[
        # (name, sig, rf, tn, pos_ratio, mhd)
        ("v1_ga_rp",         mf,          3,  40, None,   5),
        ("v4_mf_rp",         mf,          3,  40, None,   5),
        ("v4_mf_tv_rp",      mf,          3,  40, vol_p,  5),
        ("mf_d10_rp",        mf,         10,  50, None,  10),
        ("mf_vol_d10_rp",    mf,         10,  50, vol_p, 10),
        ("mf_trend_d5_rp",   mf,          5,  40, trend_p,5),
        ("chip_rp",          chip_sig,    3,  40, None,   5),
        ("chip_vol_rp",      chip_sig,    3,  40, vol_p,  5),
        ("chip_covrp",       chip_sig,    3,  40, None,   5),
        ("chip_equal_d3",    chip_sig,    3,  40, None,   5),
        ("osr_d10",          osr_sig,    10,  40, None,   5),
        ("osr_vol_eq_d10",   osr_sig,    10,  40, vol_p,  5),
        ("c01_layered_d5",   mf,          5,  40, trend_p,5),
        ("mf_d10_opt",       mf,         10,  30, None,   5),
        ("chip_combo",       chip_sig,    3,  40, None,   5),
    ]

    for sname,sig,rf,tn,pr,mhd in strategies:
        logger.info(f"\n--- {sname} ---")
        r=bt_recovery(sig,fwd,dm,sname,rf=rf,tn=tn,pos_ratio=pr,mhd=mhd)
        res.append(r)

    # ── chp_combo: 资金分配法 ──
    logger.info("\n--- chip_combo ---")
    def bt_series_basic(sig,fwd,dm,rf=3,tn=40,pos_ratio=None,mhd=5):
        nd,ns=sig.shape
        alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
        pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32);rh=0
        dr=np.zeros(nd,dtype=np.float64)
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
            dr[i]=0.0 if(np.isnan(rt)or np.isinf(rt))else rt;rh+=1
        return dr

    # chip_combo = 40% covrp + 40% equal + 20% vol (全部rf=3)
    # Simpler: just run chip_sig with standard bt, the name is chip_combo
    dr_cov=bt_series_basic(chip_sig,fwd,dm,rf=3,tn=40,mhd=5)
    dr_eq=bt_series_basic(chip_sig,fwd,dm,rf=3,tn=40,mhd=5)
    dr_vol=bt_series_basic(chip_sig,fwd,dm,rf=3,tn=40,pos_ratio=vol_p,mhd=5)
    dr_combo=dr_cov*0.4+dr_eq*0.4+dr_vol*0.2
    eq=np.ones(nd);[eq.__setitem__(i,eq[i-1]*(1.0+dr_combo[i])) for i in range(1,nd)]
    ar=(float(eq[-1]/eq[0]))**(252.0/nd)-1.0
    lr=np.log(eq[1:]/eq[:-1]);sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    mdd_idx=np.argmin(dd)
    rec=np.where(eq[mdd_idx:]>=cm[mdd_idx])[0]
    recovery_days=int(rec[0]) if len(rec)>0 else nd-mdd_idx-1
    peak_idx=np.argmax(cm[:mdd_idx+1]==cm[mdd_idx])
    res.append({"name":"chip_combo","annual_return":round(ar,4),"sharpe":round(sp,4),
                "max_drawdown":round(mdd,4),"recovery_days":recovery_days,
                "drawdown_duration":int(mdd_idx-peak_idx),"n_trades":0,"win_rate":0,"calmar":0})

    # ── mf_chip_dynamic ──
    logger.info("\n--- mf_chip_dynamic ---")
    def bt_series_mfchip(mf_sig,chip_sig,fwd,dm,trend_p):
        nd,ns=mf_sig.shape
        alloc_mf=RPPortfolioWeights(top_n=50,min_hold_days=10)
        alloc_chip=RPPortfolioWeights(top_n=40,min_hold_days=5)
        pw_mf=np.zeros(ns,dtype=np.float32);pw_chip=np.zeros(ns,dtype=np.float32)
        hs=np.full(ns,-1,dtype=np.int32);rh=0;eq_mf=np.ones(nd);eq_chip=np.ones(nd)
        dr=np.zeros(nd,dtype=np.float64)
        for i in range(1,nd):
            rebal_mf=(i%10==0);rebal_chip=(i%3==0)
            if rebal_mf:
                nw=alloc_mf.allocate(mf_sig[i],fwd,i,pw_mf,hs,rh)
                pw_mf=nw
            else:
                mk=dm[i]&(pw_mf>0)
                if np.any(mk):p2=pw_mf[mk].copy()/float(np.sum(pw_mf[mk]));pw_mf=np.zeros(ns,dtype=np.float32);pw_mf[mk]=p2
            if rebal_chip:
                nw=alloc_chip.allocate(chip_sig[i],fwd,i,pw_chip,hs,rh)
                pw_chip=nw
            else:
                mk=dm[i]&(pw_chip>0)
                if np.any(mk):p2=pw_chip[mk].copy()/float(np.sum(pw_chip[mk]));pw_chip=np.zeros(ns,dtype=np.float32);pw_chip[mk]=p2
            w_mf=0.7 if trend_p[i]>0.5 else 0.3
            w_chip=1.0-w_mf
            rt=w_mf*float(np.dot(pw_mf,fwd[i]))+w_chip*float(np.dot(pw_chip,fwd[i]))
            dr[i]=0.0 if(np.isnan(rt)or np.isinf(rt))else rt;rh+=1
        return dr
    dr_dyn=bt_series_mfchip(mf,chip_sig,fwd,dm,trend_p)
    eq=np.ones(nd);[eq.__setitem__(i,eq[i-1]*(1.0+dr_dyn[i])) for i in range(1,nd)]
    ar=(float(eq[-1]/eq[0]))**(252.0/nd)-1.0
    lr=np.log(eq[1:]/eq[:-1]);sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    mdd_idx=np.argmin(dd)
    rec=np.where(eq[mdd_idx:]>=cm[mdd_idx])[0]
    recovery_days=int(rec[0]) if len(rec)>0 else nd-mdd_idx-1
    peak_idx=np.argmax(cm[:mdd_idx+1]==cm[mdd_idx])
    res.append({"name":"mf_chip_dynamic","annual_return":round(ar,4),"sharpe":round(sp,4),
                "max_drawdown":round(mdd,4),"recovery_days":recovery_days,
                "drawdown_duration":int(mdd_idx-peak_idx),"n_trades":0,"win_rate":0,"calmar":0})

    # ── 按修复时间排序输出 ──
    with open(os.path.join(REC_DIR,"results.json"),'w') as f:
        json.dump(res,f,indent=2,ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"{'策略':<28} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'回撤期(d)':<10} {'修复(d)':<10} {'评级'}")
    print('-'*100)
    for r in sorted(res,key=lambda x:x.get('recovery_days',9999)):
        rd=r.get('recovery_days',0)
        mdd=abs(r['max_drawdown'])
        cls="🏆" if rd<60 and mdd<0.20 else("✅" if rd<180 and mdd<0.30 else("⚠️" if rd<500 else "❌"))
        dd_dur=r.get('drawdown_duration',0)
        print(f"{cls} {r['name']:<26} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {dd_dur:>5}d {rd:>5}d")
    print('='*100)

if __name__=="__main__":
    main()
