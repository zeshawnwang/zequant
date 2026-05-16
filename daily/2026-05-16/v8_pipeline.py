"""
V8 — 真实选股器信号构建 + 完整交叉实验

构建 TrendBreakout, OversoldRebound, ChipConcentration, Fundamental 在可用因子上的近似信号，
与择时器+频率全交叉。
"""
import os, sys, json, logging, gc, numpy as np, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger("v8")
TX=0.0012
FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position','beta_20']))
V8_DIR=os.path.join(os.path.dirname(__file__),"v8_results");os.makedirs(V8_DIR,exist_ok=True)

def load():
    db=Database()
    df=db.get_factors(start_date="2018-01-01",end_date="2026-04-30",factor_names=FACTORS,with_close=True)
    df['date']=pd.to_datetime(df['date']);ds=sorted(df['date'].unique())
    tks=db.get_symbols()['symbol'].tolist();nd,ns,nf=len(ds),len(tks),len(FACTORS)
    t2i={t:i for i,t in enumerate(tks)};d2i={d:i for i,d in enumerate(ds)}
    v3=np.full((nd,ns,nf),np.nan,dtype=np.float32);dm=np.zeros((nd,ns),dtype=bool)
    cl=np.zeros((nd,ns),dtype=np.float32)
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
            if len(nz)>1:
                lo,hi=np.quantile(nz,[0.01,0.99]);c=np.clip(r,lo,hi)
                mu,sd=np.mean(c),np.std(c);z3[d,:,fi]=(c-mu)/sd if sd>1e-10 else 0.0
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子")
    return z3,fwd,dm,tks,FACTORS,nd,ns,df,ds,t2i

def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

def bt(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=5)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);ttx=0.0;nt=0
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            to=float(np.sum(np.abs(nw-pw)))
            txc=0.5*to*TX;ttx+=txc
            if to>0.01:nt+=1
            pw=nw
            for j in range(ns):
                if nw[j]>0 and hs[j]<0:hs[j]=rh+1
        else:
            mk=dm[i]&(pw>0)
            if np.any(mk):
                p2=pw[mk].copy()/float(np.sum(pw[mk]))
                pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
        pr=pos_ratio[i] if pos_ratio is not None else 1.0
        rt=pr*float(np.dot(pw,fwd[i]))
        rt=0.0 if(np.isnan(rt)or np.isinf(rt))else rt
        dr[i]=rt;eq[i]=eq[i-1]*(1.0+rt);rh+=1
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

def main():
    z3,fwd,dm,tks,fnames,nd,ns,raw_df,ds,t2i=load()
    fi={fn:i for i,fn in enumerate(fnames)}
    v1w_dict=v1w()
    wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # 择时信号
    im,ims,im5,im20,ir,iv=fi.get('macd'),fi.get('macd_signal'),fi.get('momentum_5'),fi.get('momentum_20'),fi.get('rsi_14'),fi.get('volatility_20')
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
    vol_p=np.ones(nd,dtype=np.float32)
    if iv is not None:vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.05,axis=1),0.2,1.0)
    mr_p=np.full(nd,0.6,dtype=np.float32)
    if iv is not None:
        mv=np.nanmean(z3[:,:,iv],axis=1)
        mr_p[mv<0.04]=1.0;mr_p[mv>0.08]=0.3

    res=[]

    # ── 构建真实选股器代理信号 ──
    logger.info("\n=== 构建真实选股器信号 ===")

    # A02_FactorRank类: 直接用因子截面排名
    vol20_idx=fi.get('volatility_20')
    m20_idx=fi.get('momentum_20')
    rsi_idx=fi.get('rsi_14')
    vol5_idx=fi.get('beta_20')
    
    # TrendBreakout代理: a64(动量20日,高=突破)+volume_ratio(量比)+volatility_20(波动)
    tb_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    a64_idx=fi.get('a64')  # 动量20日
    volr_idx=fi.get('volume_ratio')
    for d in range(nd):
        s=np.zeros(ns)
        if a64_idx is not None:s+=z3[d,:,a64_idx]*0.5  # 动量高→突破
        if volr_idx is not None:s+=z3[d,:,volr_idx]*0.3  # 量比高→放量
        s=np.nan_to_num(s,nan=-1e10);tb_sig[d]=s

    # OversoldRebound代理: rsi低+价格跌+momentum_5负转正
    osr_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    returns_idx=fi.get('returns')
    m5_idx=fi.get('momentum_5')
    for d in range(nd):
        s=np.zeros(ns)
        if rsi_idx is not None:s+=np.where((z3[d,:,rsi_idx]<-0.5),1.0,0.0)*-0.5  # RSI低→超跌
        if returns_idx is not None:s+=np.where(z3[d,:,returns_idx]<-0.5,1.0,0.0)*0.3  # 大跌→超跌
        if m5_idx is not None:s+=np.where(z3[d,:,m5_idx]>0.3,1.0,0.0)*0.5  # 短期反弹
        s=np.nan_to_num(s,nan=-1e10);osr_sig[d]=s

    # ChipConcentration代理: 低波动+低动量(蓄势)
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None:s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5  # 低波动→筹码集中
        if m20_idx is not None:s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3  # 低动量→蓄势
        s=np.nan_to_num(s,nan=-1e10);chip_sig[d]=s

    all_sel={
        "MF":mf,"TB":tb_sig,"OSR":osr_sig,
        "Chip":chip_sig,"Vol20":np.nan_to_num(-z3[:,:,vol20_idx],nan=-1e10,neginf=-1e10) if vol20_idx is not None else mf,
        "RSI14":np.nan_to_num(z3[:,:,rsi_idx],nan=-1e10,neginf=-1e10) if rsi_idx is not None else mf,
    }
    all_pos={"无择时":None,"Trend":trend_p,"Vol":vol_p,"MR":mr_p}
    all_freq=[("D3",3),("D5",5),("D10",10)]

    # ── 全部选股器 × 择时器 × 频率 ──
    logger.info("\n=== 全部选股器×择时器×频率 ===")
    for sname,ssig in all_sel.items():
        for pname,ppos in all_pos.items():
            for fname,fd in all_freq:
                label=f"{sname}_{pname}_{fname}"
                res.append(bt(ssig,fwd,dm,label,rf=fd,pos_ratio=ppos))

    # ── C系列: 三变量组合 ──
    logger.info("\n=== C系列: 三变量组合 ===")
    # C01: MF+Trend+D3 (已经跑过)
    # C02: MF+MR+D5+VolP(真风险平价)
    res.append(bt(mf,fwd,dm,"C02_MF+MR+D5",rf=5,pos_ratio=mr_p))
    # C03: TB+Trend+D5
    res.append(bt(tb_sig,fwd,dm,"C03_TB+Trend+D5",rf=5,pos_ratio=trend_p))
    # C04: OSR+Vol+D10
    res.append(bt(osr_sig,fwd,dm,"C04_OSR+Vol+D10",rf=10,pos_ratio=vol_p))
    # C05: Chip+MR+D10
    res.append(bt(chip_sig,fwd,dm,"C05_Chip+MR+D10",rf=10,pos_ratio=mr_p))
    # C06~10: 已完成或覆盖率足够

    # ── D系列: 完整策略链路 ──
    logger.info("\n=== D系列: 完整策略 ===")
    res.append(bt(tb_sig,fwd,dm,"D03_TB+Trend+D3+RP",rf=3,pos_ratio=trend_p))

    # ── 保存 ──
    with open(os.path.join(V8_DIR,"results.json"),'w') as f:
        json.dump(res,f,indent=2,ensure_ascii=False)

    print(f"\n{'='*110}")
    print(f"{'实验':<32} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'换手':<5}")
    print('-'*110)
    for r in sorted(res,key=lambda x:x['sharpe'],reverse=True):
        ok=r['max_drawdown']<0.25 and r['annual_return']>0.10
        cls="🏆" if ok else("  ")
        print(f"{cls} {r['name']:<30} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")
    print('='*110)

    logger.info("\n达标(回撤<25%且年化>10%):")
    for r in sorted(res,key=lambda x:x['sharpe'],reverse=True):
        if r['max_drawdown']<0.25 and r['annual_return']>0.10:
            logger.info(f"  🏆 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")

if __name__=="__main__":
    main()
