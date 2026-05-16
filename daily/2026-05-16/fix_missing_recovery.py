"""补跑缺失策略的回撤修复时间 + 输出完整数据表"""
import os,sys,json,logging,numpy as np,pandas as pd

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
logger=logging.getLogger("fix_recovery")
TX=0.0012

FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5','momentum_20',
    'volume_ratio','boll_position','beta_20']))
NEW=['ma5','ma10','ma20','ma21','ma60','ma120','ma_alignment_score','ma60_trend',
    'ma120_trend','macd_above_zero','macd_golden_cross','volume_breakout_ratio',
    'volume_contraction','ma_convergence','chip_concentration','ma_angle_20']
ALL=list(set(FACTORS+NEW))

def load():
    db=Database()
    ac=[c for c in ALL if c in db.list_factor_columns()]
    df=db.get_factors(start_date="2018-01-01",end_date="2026-04-30",factor_names=ac,with_close=True)
    df['date']=pd.to_datetime(df['date']);ds=sorted(df['date'].unique())
    tks=db.get_symbols()['symbol'].tolist();nd,ns,nf=len(ds),len(tks),len(ac)
    t2i={t:i for i,t in enumerate(tks)};d2i={d:i for i,d in enumerate(ds)}
    v3=np.full((nd,ns,nf),np.nan,dtype=np.float32);dm=np.zeros((nd,ns),dtype=bool);cl=np.zeros((nd,ns),dtype=np.float32)
    di=np.array([d2i[d] for d in df['date']],dtype=np.int32)
    si=np.array([t2i.get(s,-1) for s in df['symbol']],dtype=np.int32)
    v=si>=0;di,si=di[v],si[v]
    for fi,fc in enumerate(ac):
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
    return z3,fwd,dm,tks,ac,nd,ns,ds

def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

def load_ga_weights():
    p=os.path.join(os.path.dirname(__file__),'..','..','daily','2026-05-16','x4_x5_results','x5_results.json')
    if os.path.exists(p):
        with open(p) as f:
            data=json.load(f)
            if isinstance(data,list) and len(data)>0:
                if 'weights' in data[0]:
                    return {k:float(v) for k,v in data[0]['weights'].items()}
    return {}

def calc_metrics(eq,nd,name):
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    mdd_idx=np.argmin(dd)
    rec=np.where(eq[mdd_idx:]>=cm[mdd_idx])[0]
    rd=int(rec[0]) if len(rec)>0 else nd-mdd_idx-1
    pk_idx=np.argmax(cm[:mdd_idx+1]==cm[mdd_idx])
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    return {"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),
            "calmar":round(cal,4),"recovery_days":rd,"drawdown_duration":int(mdd_idx-pk_idx)}

def bt_full(sig,fwd,dm,rf=3,tn=40,pr=None,mhd=5):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32);rh=0
    eq=np.ones(nd,dtype=np.float64)
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr_i=pr[i] if pr is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):p2=pw[mk].copy()/float(np.sum(pw[mk]));pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr_i=pr[i] if pr is not None else 1.0
        rt=pr_i*float(np.dot(pw,fwd[i]))
        rt=0.0 if(np.isnan(rt)or np.isinf(rt))else rt
        eq[i]=eq[i-1]*(1.0+rt);rh+=1
    return eq

def bt_full_series(sig,fwd,dm,rf=3,tn=40,pr=None,mhd=5):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32);rh=0
    dr=np.zeros(nd,dtype=np.float64)
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr_i=pr[i] if pr is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):p2=pw[mk].copy()/float(np.sum(pw[mk]));pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr_i=pr[i] if pr is not None else 1.0
        rt=pr_i*float(np.dot(pw,fwd[i]))
        dr[i]=0.0 if(np.isnan(rt)or np.isinf(rt))else rt;rh+=1
    return dr

def main():
    z3,fwd,dm,tks,fnames,nd,ns,ds=load()
    fi={fn:i for i,fn in enumerate(fnames)}
    v1w_dict=v1w()
    ga_dict=load_ga_weights()
    
    # MF
    wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # GA
    gaw=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in ga_dict:gaw[fi_i]=float(ga_dict[fc])
    s2=np.sum(np.abs(gaw));gaw/=s2 if s2>0 else 1
    ga=np.nan_to_num(np.tensordot(z3,gaw,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # Chip
    vi=fi.get('volatility_20');mi=fi.get('momentum_20')
    chip=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s_chip=np.zeros(ns)
        if vi is not None:s_chip+=np.where(z3[d,:,vi]<-0.3,1.0,0.0)*0.5
        if mi is not None:s_chip+=np.where(np.abs(z3[d,:,mi])<0.3,1.0,0.0)*0.3
        chip[d]=np.nan_to_num(s_chip,nan=-1e10)

    # 择时
    iv=fi.get('volatility_20')
    vol_p=np.ones(nd,dtype=np.float32)
    if iv is not None:vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.05,axis=1),0.2,1.0)
    # ht=0.25版本
    vol_p25=np.ones(nd,dtype=np.float32)
    if iv is not None:vol_p25=np.clip(1.0-np.mean(z3[:,:,iv]>0.25,axis=1),0.2,1.0)

    res=[]

    # ── 缺失的7个策略 ──
    missing=[
        ("ga_d5", ga, 5, 40, None, 5),
        ("ga_d10", ga, 10, 40, None, 5),
        ("mf_vol_d10_opt", mf, 10, 30, vol_p25, 5),
    ]
    for name,sig,rf,tn,pr,mhd in missing:
        logger.info(f"--- {name} ---")
        eq=bt_full(sig,fwd,dm,rf=rf,tn=tn,pr=pr,mhd=mhd)
        m=calc_metrics(eq,nd,name)
        m["name"]=name;res.append(m)

    # ── 组合策略 ──
    combo_defs=[
        ("mf50_chip50_combo", 0.5, 0.5, 10, 3, 50, 40, None, None),
        ("mf50_chipcovrp50_combo", 0.5, 0.5, 10, 3, 50, 40, None, None),
        ("mf60_chip40_combo", 0.6, 0.4, 10, 3, 50, 40, None, None),
        ("ga_covrp_combo", 0.6, 0.4, 10, 3, 40, 40, None, None),
    ]
    for name,w1,w2,rf1,rf2,tn1,tn2,pr1,pr2 in combo_defs:
        logger.info(f"--- {name} ---")
        dr1=bt_full_series(mf,fwd,dm,rf=rf1,tn=tn1,pr=pr1 if pr1 is not None and isinstance(pr1,np.ndarray) else None,mhd=10)
        dr2=bt_full_series(chip,fwd,dm,rf=rf2,tn=tn2,pr=pr2 if pr2 is not None and isinstance(pr2,np.ndarray) else None,mhd=5)
        dr=dr1*w1+dr2*w2
        eq=np.ones(nd)
        for i in range(1,nd):eq[i]=eq[i-1]*(1.0+dr[i])
        m=calc_metrics(eq,nd,name)
        m["name"]=name;res.append(m)

    # ── 输出 ──
    with open(os.path.join(os.path.dirname(__file__),"recovery_results","missing_results.json"),'w') as f:
        json.dump(res,f,indent=2,ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"{'策略':<28} {'年化%':<8} {'Sharpe':<8} {'最大回撤%':<10} {'回撤期(d)':<10} {'修复(d)':<10}")
    print('-'*100)
    for r in sorted(res,key=lambda x:x['recovery_days']):
        print(f"  {r['name']:<26} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['drawdown_duration']:>5}d {r['recovery_days']:>5}d")
    print('='*100)

if __name__=="__main__":
    main()
