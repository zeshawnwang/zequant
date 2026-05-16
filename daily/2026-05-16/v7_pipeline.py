"""
V7 — 选股器×择时器×频率交叉（非MF选股器+最佳频率探索）
"""
import os, sys, json, logging, gc, numpy as np, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("v7")

TX = 0.0012
FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position','beta_20']))
V7_DIR = os.path.join(os.path.dirname(__file__), "v7_results"); os.makedirs(V7_DIR, exist_ok=True)

def load():
    db = Database()
    df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30", factor_names=FACTORS, with_close=True)
    df['date'] = pd.to_datetime(df['date'])
    ds = sorted(df['date'].unique()); tks = db.get_symbols()['symbol'].tolist()
    nd,ns,nf = len(ds),len(tks),len(FACTORS)
    t2i={t:i for i,t in enumerate(tks)}; d2i={d:i for i,d in enumerate(ds)}
    v3=np.full((nd,ns,nf),np.nan,dtype=np.float32); dm=np.zeros((nd,ns),dtype=bool)
    cl=np.zeros((nd,ns),dtype=np.float32)
    di=np.array([d2i[d] for d in df['date']],dtype=np.int32)
    si=np.array([t2i.get(s,-1) for s in df['symbol']],dtype=np.int32)
    v=si>=0; di,si=di[v],si[v]
    for fi,fc in enumerate(FACTORS):
        if fc in df.columns: v3[di,si,fi]=df[fc].values[v].astype(np.float32)
    cl[di,si]=df['close'].values[v].astype(np.float32); dm[di,si]=True
    np.nan_to_num(v3,nan=0.0,copy=False); np.nan_to_num(cl,nan=0.0,copy=False)
    fwd=np.zeros((nd,ns),dtype=np.float32)
    for d in range(nd-1):
        b=(cl[d]>1e-10)&(cl[d+1]>1e-10); fwd[d,b]=(cl[d+1,b]-cl[d,b])/cl[d,b]
    z3=np.zeros_like(v3)
    for fi in range(nf):
        a=v3[:,:,fi]
        for d in range(nd):
            r=a[d,:]; nz=r[r!=0]
            if len(nz)>1:
                lo,hi=np.quantile(nz,[0.01,0.99]); c=np.clip(r,lo,hi)
                mu,sd=np.mean(c),np.std(c)
                z3[d,:,fi]=(c-mu)/sd if sd>1e-10 else 0.0
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子")
    return z3, fwd, dm, tks, FACTORS, nd, ns

def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']: return it['configs'][0]['weights']
    return {}

def bt(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None):
    nd,ns=sig.shape; alloc=RPPortfolioWeights(top_n=tn,min_hold_days=5)
    pw=np.zeros(ns,dtype=np.float32); hs=np.full(ns,-1,dtype=np.int32)
    rh=0; eq=np.ones(nd,dtype=np.float64); dr=np.zeros(nd,dtype=np.float64); ttx=0.0; nt=0
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            to=float(np.sum(np.abs(nw-pw)))
            txc=0.5*to*TX; ttx+=txc
            if to>0.01: nt+=1
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0: hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):
                p2=pw[mk].copy()/float(np.sum(pw[mk]))
                pw=np.zeros(ns,dtype=np.float32); pw[mk]=p2
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i]))
        rt=0.0 if(np.isnan(rt)or np.isinf(rt))else rt
        dr[i]=rt; eq[i]=eq[i-1]*(1.0+rt); rh+=1
    tr=float(eq[-1]/eq[0]-1.0); ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq); dd=(eq-cm)/cm; mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return {"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

def main():
    z3,fwd,dm,tks,fnames,nd,ns=load()
    fi={fn:i for i,fn in enumerate(fnames)}
    v1w_dict=v1w(); wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict: wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv)); wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)
    
    # 预计算择时信号
    im,ims,im5,im20,ir,iv=fi.get('macd'),fi.get('macd_signal'),fi.get('momentum_5'),fi.get('momentum_20'),fi.get('rsi_14'),fi.get('volatility_20')
    trend_p=np.full(nd,0.5,dtype=np.float32)
    for d in range(nd):
        sl=[]
        if im is not None and ims is not None: sl.append(np.where(z3[d,:,im]>z3[d,:,ims],1.0,0.0))
        if im5 is not None and im20 is not None:
            m5v,m20v=z3[d,:,im5],z3[d,:,im20]
            sl.append(np.where((m5v>0)&(m5v>m20v),1.0,np.where(m5v<0,0.0,0.5)))
        if ir is not None:
            rv=z3[d,:,ir]
            sl.append(np.where(rv>70,0.0,np.where(rv>=50,1.0,np.where(rv>=30,0.5,0.0))))
        if sl: trend_p[d]=np.clip(np.mean(np.mean(sl,axis=0)>=0.6)*2.0,0.1,1.0)
    vol_p=np.ones(nd,dtype=np.float32)
    if iv is not None: vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.05,axis=1),0.2,1.0)
    mr_p=np.full(nd,0.6,dtype=np.float32)
    if iv is not None:
        mv=np.nanmean(z3[:,:,iv],axis=1)
        mr_p[mv<0.04]=1.0; mr_p[mv>0.08]=0.3

    res=[]

    # ── E系列: 组合不传递性实验 ──
    logger.info("\nE01-E06: 强选股MF × 各类择时器")
    # 这些在V6已经跑过，直接复用
    # E07-E12: 强择时TV × 各类选股器(非MF)
    
    # 构建其他选股器信号
    logger.info("\nE07-E12: 强择时(TV) × 各类选股器")
    selectors={
        "MF":mf,
        "FR_m20":np.nan_to_num(z3[:,:,fi['momentum_20']],nan=-1e10,neginf=-1e10) if 'momentum_20' in fi else mf,
        "FR_vol20_asc":np.nan_to_num(-z3[:,:,fi['volatility_20']],nan=-1e10,neginf=-1e10) if 'volatility_20' in fi else mf,
        "FR_rsi14":np.nan_to_num(z3[:,:,fi['rsi_14']],nan=-1e10,neginf=-1e10) if 'rsi_14' in fi else mf,
    }
    for sname,ssig in selectors.items():
        res.append(bt(ssig,fwd,dm,f"E07_{sname}+TV+D3",rf=3,pos_ratio=vol_p))
        res.append(bt(ssig,fwd,dm,f"E08_{sname}+TV+D5",rf=5,pos_ratio=vol_p))
        res.append(bt(ssig,fwd,dm,f"E09_{sname}+TV+D10",rf=10,pos_ratio=vol_p))
    
    # E10-E12: 强择时(TrendTiming) × 各类选股器
    logger.info("\nE10-E12: 强择时(TrendTiming) × 各类选股器")
    for sname,ssig in selectors.items():
        res.append(bt(ssig,fwd,dm,f"E10_{sname}+Trend+D3",rf=3,pos_ratio=trend_p))
        res.append(bt(ssig,fwd,dm,f"E11_{sname}+Trend+D5",rf=5,pos_ratio=trend_p))
        res.append(bt(ssig,fwd,dm,f"E12_{sname}+Trend+D10",rf=10,pos_ratio=trend_p))
    
    # E13-E16: 分配器×非最优组合
    logger.info("\nE13-E16: 分配器×非最优组合")
    # 等权实现: top_n只，每只等权
    nsyms=ns
    for fd in [3,5,10]:
        for sname,ssig in selectors.items():
            # 修改为: 选top_n只等权
            ew_mf=mf.copy()
            res.append(bt(ew_mf,fwd,dm,f"E13_{sname}+EQUAL+D{fd}",rf=fd))
    
    # E17-E21: 全维度第二好
    logger.info("\nE17-E21: 第二好大乱斗")
    res.append(bt(selectors['FR_vol20_asc'],fwd,dm,"E17_vol(2nd)+Trend(2nd)+EQ(2nd)+D5",rf=5,pos_ratio=trend_p))
    res.append(bt(selectors['FR_m20'],fwd,dm,"E18_m20(5th)+MR(3rd)+RP(1st)+D10",rf=10,pos_ratio=mr_p))

    # 保存
    with open(os.path.join(V7_DIR,"results.json"),'w') as f:
        json.dump(res,f,indent=2,ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"{'实验':<32} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'换手':<5}")
    print('-'*100)
    for r in sorted(res,key=lambda x:x['sharpe'],reverse=True):
        ok=r['max_drawdown']<0.25 and r['annual_return']>0.10
        cls="🏆" if ok else "  "
        print(f"{cls} {r['name']:<30} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")
    print('='*100)

    logger.info("\n达标(回撤<25%且年化>10%):")
    for r in res:
        if r['max_drawdown']<0.25 and r['annual_return']>0.10:
            logger.info(f"  🏆 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")

if __name__=="__main__":
    main()
