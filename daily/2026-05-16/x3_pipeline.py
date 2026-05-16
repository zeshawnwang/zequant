"""
X3 — 真实选股器信号构建 + Type E剩余实验

用实时计算的SMA(ma5/ma20/ma60/ma120/ma21/ma99)构建真实选股器信号，
与V6一致的择时信号做全交叉实验 + Type E剩余实验。

构建的信号:
  - TrendBreakout真实: 均线多头+放量+MACD>0
  - OversoldRebound真实: MACD形态+价格位置+均线角度
  - ChipConcentration真实: 量缩+低波动+筹码集中
"""
import os,sys,json,logging,gc,numpy as np,pandas as pd
from datetime import datetime

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger("x3")
TX=0.0012

# ──────────────────────────────────────────
# 因子列表
# ──────────────────────────────────────────
FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position','beta_20']))

X3_DIR=os.path.join(os.path.dirname(__file__),"x3_results");os.makedirs(X3_DIR,exist_ok=True)

# ──────────────────────────────────────────
# 数据加载 (返回 v3原始值 + z3截面Z-Score + cl收盘价)
# ──────────────────────────────────────────
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
    fi={fn:i for i,fn in enumerate(FACTORS)}
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子")
    return z3,v3,fwd,dm,cl,tks,FACTORS,nd,ns,ds,t2i,fi

# ──────────────────────────────────────────
# V1权重加载
# ──────────────────────────────────────────
def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

# ──────────────────────────────────────────
# SMA计算 (简单移动平均)
# ──────────────────────────────────────────
def compute_sma(cl, w):
    nd,ns=cl.shape;out=np.zeros_like(cl)
    for d in range(nd):
        if d<w-1:
            out[d]=np.mean(cl[:d+1],axis=0) if d>0 else cl[0]
        else:
            out[d]=np.mean(cl[d-w+1:d+1],axis=0)
    return out

# ──────────────────────────────────────────
# 回测函数: RP分配 (同V8)
# ──────────────────────────────────────────
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

# ──────────────────────────────────────────
# 回测函数: 等权分配
# ──────────────────────────────────────────
def bt_ew(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None):
    nd,ns=sig.shape
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);nt=0
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            si=np.argsort(-sig[i])[:tn]
            nw=np.zeros(ns)
            if len(si)>0:
                ew=pr/len(si)
                nw[si]=ew
            to=float(np.sum(np.abs(nw-pw)))
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
    ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

# ──────────────────────────────────────────
# 回测函数: 协方差风险平价 (CovRP)
# ──────────────────────────────────────────
def bt_covrp(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None):
    nd,ns=sig.shape
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);nt=0
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            si=np.argsort(-sig[i])[:tn]
            if i>=20:
                seg=fwd[max(0,i-20):i,:]
                sub=seg[:,si]
                sub=sub[:,~np.any(np.isnan(sub)|np.isinf(sub),axis=0)]
                if sub.shape[1]>=2:
                    try:
                        cov=np.cov(sub.T)
                        iv=1.0/np.sqrt(np.diag(cov)+1e-10)
                    except:
                        iv=np.ones(sub.shape[1])
                else:
                    iv=np.ones(sub.shape[1])
            else:
                iv=np.ones(min(tn,ns))
            nw=np.zeros(ns)
            sidx=si[:len(iv)]
            if len(sidx)>0:
                nw[sidx]=iv/np.sum(iv)
            to=float(np.sum(np.abs(nw-pw)))
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
    ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

# ──────────────────────────────────────────
# 回测函数: 迟滞分配器 (HysteresisAllocator)
# 目标仓位 = 当前仓位 + (新仓位 - 当前仓位) * hysteresis_rate
# ──────────────────────────────────────────
def bt_hysteresis(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None,hysteresis_rate=0.5):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=5)
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);ttx=0.0;nt=0
    for i in range(1,nd):
        rebal=(i%rf==0)
        if rebal:
            pr=pos_ratio[i] if pos_ratio is not None else 1.0
            nw_target=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
            # 迟滞: 从当前仓位向目标仓位移动
            nw_current=np.zeros(ns,dtype=np.float32)
            for j in range(ns):
                if pw[j]>0:
                    pk=list(pw.nonzero()[0])
                    if j in pk:
                        nw_current[j]=pw[j]/np.sum(pw[pk])*pr
            nw=nw_current+(nw_target-nw_current)*hysteresis_rate
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
    ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

# ──────────────────────────────────────────
# 择时信号构造 (同V6/V7/V8)
# ──────────────────────────────────────────
def build_timing_signals(z3, fi, nd):
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

    return trend_p, vol_p, mr_p

# ──────────────────────────────────────────
# 真实选股器信号构造
# ──────────────────────────────────────────
def build_real_signals(z3, v3, cl, fi, nd, ns):
    """构建3个真实选股器信号 + MF信号."""
    # 计算SMA
    sma5=compute_sma(cl,5)
    sma20=compute_sma(cl,20)
    sma60=compute_sma(cl,60)
    sma120=compute_sma(cl,120)
    sma21=compute_sma(cl,21)
    sma99=compute_sma(cl,99)

    # ── 1. TrendBreakout 真实信号 ──
    # filter: sma5>sma20 AND sma20>sma60 AND close>sma60 AND macd>0 AND volume_ratio>1.5
    # score: volume_ratio排序
    logger.info("构建 TrendBreakout 真实信号...")
    tb_real=np.full((nd,ns),-np.inf,dtype=np.float32)
    macd_idx=fi.get('macd')
    volr_idx=fi.get('volume_ratio')
    for d in range(nd):
        f5=sma5[d]>sma20[d]
        f20=sma20[d]>sma60[d]
        fc=cl[d]>sma60[d]
        fm=z3[d,:,macd_idx]>0 if macd_idx is not None else np.ones(ns)
        fv=(v3[d,:,volr_idx]>1.5) if volr_idx is not None else np.ones(ns)
        filt=f5&f20&fc&fm&fv
        # score: volume_ratio Z-score排序
        sc=np.full(ns,-1e10,dtype=np.float32)
        if volr_idx is not None:
            sc[filt]=z3[d,filt,volr_idx]
        else:
            sc[filt]=1.0
        tb_real[d]=sc

    # ── 2. OversoldRebound 真实信号 ──
    # score: macd_arc_bottom*2 + macd_golden_cross*2 + macd<0 
    #   + close>ma21 + espacio(ma99-close)/close + ma_angle_20>-2 + macd>macd_signal
    logger.info("构建 OversoldRebound 真实信号...")
    osr_real=np.full((nd,ns),-np.inf,dtype=np.float32)
    macdsig_idx=fi.get('macd_signal')
    for d in range(nd):
        # macd_golden_cross: macd[d] > macd_signal[d] 且前一天 macd[d-1] <= macd_signal[d-1]
        golden=np.zeros(ns,dtype=bool)
        if d>=1 and macd_idx is not None and macdsig_idx is not None:
            golden=(z3[d,:,macd_idx]>z3[d,:,macdsig_idx])&(z3[d-1,:,macd_idx]<=z3[d-1,:,macdsig_idx])

        # macd_arc_bottom: 连续2+日macd上升, 且日前处于低位
        arc_bottom=np.zeros(ns,dtype=bool)
        if d>=3 and macd_idx is not None:
            arc_bottom=(z3[d,:,macd_idx]>z3[d-1,:,macd_idx])&(z3[d-1,:,macd_idx]>z3[d-2,:,macd_idx])

        # macd<0
        macd_neg=np.zeros(ns,dtype=bool)
        if macd_idx is not None:
            macd_neg=z3[d,:,macd_idx]<0

        # close>ma21
        close_above_ma21=cl[d]>sma21[d]

        # espacio(ma99-close)/close
        espacio=(sma99[d]-cl[d])/np.clip(cl[d],1e-10,None)
        # normalize espacio to Z-score for scoring
        nz=espacio[cl[d]>1e-10]
        if len(nz)>1:
            lo,hi=np.quantile(nz,[0.01,0.99]);ec=np.clip(espacio,lo,hi)
            mu,sd=np.mean(ec),np.std(ec)
            espacio_z=(ec-mu)/sd if sd>1e-10 else np.zeros(ns)
        else:
            espacio_z=np.zeros(ns)

        # ma_angle_20 > -2° : sma20斜率 > -0.035 (tan(-2°)≈-0.035)
        angle_ok=np.ones(ns,dtype=bool)
        if d>=5:
            slope=(sma20[d]-sma20[d-5])/np.clip(sma20[d],1e-10,None)
            angle_ok=slope>-0.035

        # macd>macd_signal
        macd_above_signal=np.zeros(ns,dtype=bool)
        if macd_idx is not None and macdsig_idx is not None:
            macd_above_signal=z3[d,:,macd_idx]>z3[d,:,macdsig_idx]

        sc=np.zeros(ns)
        sc+=arc_bottom.astype(float)*2.0
        sc+=golden.astype(float)*2.0
        sc+=macd_neg.astype(float)*1.0
        sc+=close_above_ma21.astype(float)*1.0
        sc+=np.clip(espacio_z,0,None)*1.0  # only positive space counts
        sc+=angle_ok.astype(float)*1.0
        sc+=macd_above_signal.astype(float)*1.0
        osr_real[d]=sc

    # ── 3. ChipConcentration 真实信号 ──
    # filter: volume_ratio<0.5(量缩) AND volatility_20<中位数(低波动)
    # score: (1-volume_ratio_norm) + (1-volatility_20_norm) + breakout_strength
    logger.info("构建 ChipConcentration 真实信号...")
    chip_real=np.full((nd,ns),-np.inf,dtype=np.float32)
    vol20_idx=fi.get('volatility_20')
    for d in range(nd):
        # filter (raw values)
        vr_lo=np.ones(ns,dtype=bool)
        if volr_idx is not None:
            vr_lo=v3[d,:,volr_idx]<0.5
        vol_lo=np.ones(ns,dtype=bool)
        if vol20_idx is not None:
            vol_med=np.nanmedian(v3[d,:,vol20_idx])
            vol_lo=v3[d,:,vol20_idx]<vol_med
        filt=vr_lo&vol_lo

        # score components (Z-score)
        sc=np.full(ns,-1e10,dtype=np.float32)
        if np.any(filt):
            sc1=np.zeros(ns)  # 1-volume_ratio_norm
            if volr_idx is not None:
                vr_z=z3[d,:,volr_idx]
                vr_norm=(vr_z-np.min(vr_z))/(np.max(vr_z)-np.min(vr_z)+1e-10)
                sc1=1.0-vr_norm
            sc2=np.zeros(ns)  # 1-volatility_norm
            if vol20_idx is not None:
                vol_z=z3[d,:,vol20_idx]
                vol_norm=(vol_z-np.min(vol_z))/(np.max(vol_z)-np.min(vol_z)+1e-10)
                sc2=1.0-vol_norm
            # breakout_strength: sma5与sma20的比值 (接近=筹码集中)
            sc3=1.0-np.abs(sma5[d]/np.clip(sma20[d],1e-10,None)-1.0)*10
            sc3=np.clip(sc3,0,1)

            sc[filt]=sc1[filt]+sc2[filt]+sc3[filt]
        chip_real[d]=sc

    # ── 4. MF多因子信号 ──
    logger.info("构建 MF 信号...")
    v1w_dict=v1w()
    wv=np.zeros(len(FACTORS),dtype=np.float32)
    for fi_i,fc in enumerate(FACTORS):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # ── 5. vol20_asc信号 (volatility_20小到大 = 低波动优先) ──
    vol20_asc=np.full((nd,ns),-np.inf,dtype=np.float32)
    if vol20_idx is not None:
        for d in range(nd):
            sc=-z3[d,:,vol20_idx]  # 负的Z-score, 低波动=高排名
            vol20_asc[d]=sc

    return tb_real, osr_real, chip_real, mf, vol20_asc

# ──────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────
def main():
    logger.info("="*60)
    logger.info("X3 — 真实选股器信号 + Type E剩余实验")
    logger.info("="*60)

    # 1. 加载数据
    z3,v3,fwd,dm,cl,tks,fnames,nd,ns,ds,t2i,fi=load()

    # 2. 构建择时信号
    logger.info("\n=== 构建择时信号 ===")
    trend_p,vol_p,mr_p=build_timing_signals(z3,fi,nd)
    logger.info(f"  trend_p: mean={trend_p.mean():.3f} [0.1~1.0]")
    logger.info(f"  vol_p:   mean={vol_p.mean():.3f} [0.2~1.0]")
    logger.info(f"  mr_p:    mean={mr_p.mean():.3f} [0.3~1.0]")

    # 3. 构建真实选股器信号
    logger.info("\n=== 构建真实选股器信号 ===")
    tb_real,osr_real,chip_real,mf,vol20_asc=build_real_signals(z3,v3,cl,fi,nd,ns)

    all_sel={
        "TB_real":tb_real,"OSR_real":osr_real,"Chip_real":chip_real,
        "MF":mf,"Vol20_asc":vol20_asc,
    }

    results=[]

    # ── 4. 真实信号 × 择时方式 × 频率 (基准交叉) ──
    logger.info("\n"+"="*60)
    logger.info("Step 4: 真实信号 × 择时方式 × 频率")
    logger.info("="*60)
    all_pos={"无择时":None,"VolTiming":vol_p,"TrendTiming":trend_p,"MarketRegime":mr_p}
    all_freq=[("D3",3),("D5",5),("D10",10)]
    real_names=["TB_real","OSR_real","Chip_real"]
    for sname in real_names:
        ssig=all_sel[sname]
        for pname,ppos in all_pos.items():
            for fname,fd in all_freq:
                label=f"{sname}_{pname}_{fname}"
                results.append(bt(ssig,fwd,dm,label,rf=fd,pos_ratio=ppos))

    # ── 5. Type E剩余实验 ──
    logger.info("\n"+"="*60)
    logger.info("Step 5: Type E剩余实验")
    logger.info("="*60)

    # E09: TrendBreakout真实 + TVTiming(vol_p) + D3/D5/D10
    logger.info("\n--- E09: TB_real + TVTiming(vol_p) ---")
    for fname,fd in [("D3",3),("D5",5),("D10",10)]:
        results.append(bt(tb_real,fwd,dm,f"E09_TB_real+TVTiming_{fname}",rf=fd,pos_ratio=vol_p))

    # E10: OversoldRebound真实 + TVTiming(vol_p) + D3/D5/D10
    logger.info("\n--- E10: OSR_real + TVTiming(vol_p) ---")
    for fname,fd in [("D3",3),("D5",5),("D10",10)]:
        results.append(bt(osr_real,fwd,dm,f"E10_OSR_real+TVTiming_{fname}",rf=fd,pos_ratio=vol_p))

    # E11: ChipConcentration真实 + TVTiming(vol_p) + D3/D5/D10
    logger.info("\n--- E11: Chip_real + TVTiming(vol_p) ---")
    for fname,fd in [("D3",3),("D5",5),("D10",10)]:
        results.append(bt(chip_real,fwd,dm,f"E11_Chip_real+TVTiming_{fname}",rf=fd,pos_ratio=vol_p))

    # E13: MF(排位) + TrendTiming(trend_p) + EqualWeight + D3/D5
    logger.info("\n--- E13: MF + TrendTiming + EqualWeight ---")
    for fname,fd in [("D3",3),("D5",5)]:
        results.append(bt_ew(mf,fwd,dm,f"E13_MF+TrendTiming+EQ_{fname}",rf=fd,pos_ratio=trend_p))

    # E15: TrendBreakout真实 + TrendTiming(trend_p) + EqualWeight + D5
    logger.info("\n--- E15: TB_real + TrendTiming + EQ + D5 ---")
    results.append(bt_ew(tb_real,fwd,dm,"E15_TB_real+TrendTiming+EQ_D5",rf=5,pos_ratio=trend_p))

    # E16: OversoldRebound真实 + VolTiming(vol_p) + EqualWeight + D10
    logger.info("\n--- E16: OSR_real + VolTiming + EQ + D10 ---")
    results.append(bt_ew(osr_real,fwd,dm,"E16_OSR_real+VolTiming+EQ_D10",rf=10,pos_ratio=vol_p))

    # E17: vol20_asc选股 + TrendTiming(trend_p) + EqualWeight + D5 (验证)
    logger.info("\n--- E17: Vol20_asc + TrendTiming + EQ + D5 (验证) ---")
    results.append(bt_ew(vol20_asc,fwd,dm,"E17_vol20_asc+TrendTiming+EQ_D5",rf=5,pos_ratio=trend_p))

    # E18: TrendBreakout真实 + MarketRegime(mr_p) + RiskParity(cov) + D10
    logger.info("\n--- E18: TB_real + MR + CovRP + D10 ---")
    results.append(bt_covrp(tb_real,fwd,dm,"E18_TB_real+MR+CovRP_D10",rf=10,pos_ratio=mr_p))

    # E19: ChipConcentration真实 + VolTiming(vol_p) + RP(cov) + D3
    logger.info("\n--- E19: Chip_real + VolTiming + CovRP + D3 ---")
    results.append(bt_covrp(chip_real,fwd,dm,"E19_Chip_real+VolTiming+CovRP_D3",rf=3,pos_ratio=vol_p))

    # E20: OversoldRebound真实 + TrendTiming(trend_p) + TrendSizer + D3
    # TrendSizer: 使用趋势仓位系数 (与bt中pos_ratio=trend_p等价)
    logger.info("\n--- E20: OSR_real + TrendTiming + TrendSizer + D3 ---")
    results.append(bt(osr_real,fwd,dm,"E20_OSR_real+TrendTiming+TrendSizer_D3",rf=3,pos_ratio=trend_p))

    # E21: MF + 无择时 + HysteresisAllocator(enable=true) + D3
    logger.info("\n--- E21: MF + 无择时 + Hysteresis + D3 ---")
    results.append(bt_hysteresis(mf,fwd,dm,"E21_MF+无择时+Hysteresis_D3",rf=3,hysteresis_rate=0.5))

    # ── 6. 保存结果 ──
    with open(os.path.join(X3_DIR,"results.json"),'w') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    # ── 7. 输出汇总 ──
    print(f"\n{'='*120}")
    print(f"{'实验名称':<36} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'换手':<5}")
    print('-'*120)

    # 按类型分组
    step4_results=[r for r in results if not any(r['name'].startswith(e) for e in ['E09','E10','E11','E13','E15','E16','E17','E18','E19','E20','E21'])]
    type_e_results=[r for r in results if any(r['name'].startswith(e) for e in ['E09','E10','E11','E13','E15','E16','E17','E18','E19','E20','E21'])]

    print(f"\n--- Step 4: 真实信号×择时×频率 (共{len(step4_results)}个实验) ---")
    for r in sorted(step4_results,key=lambda x:x['sharpe'],reverse=True):
        ok=r['max_drawdown']<0.25 and r['annual_return']>0.10
        cls="🏆" if ok else "  "
        print(f"{cls} {r['name']:<34} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")

    print(f"\n--- Type E剩余实验 (共{len(type_e_results)}个实验) ---")
    for r in sorted(type_e_results,key=lambda x:x['sharpe'],reverse=True):
        ok=r['max_drawdown']<0.25 and r['annual_return']>0.10
        cls="🏆" if ok else "  "
        print(f"{cls} {r['name']:<34} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")

    print('='*120)

    # 汇总
    logger.info("\n=== 真实信号 vs V8代理信号对比 ===")
    # 加载V8结果
    v8_path=os.path.join(os.path.dirname(__file__),"v8_results","results.json")
    if os.path.exists(v8_path):
        with open(v8_path) as f:
            v8_results=json.load(f)
        v8_map={r['name']:r for r in v8_results}
        # 对比相同配置下真实信号 vs 代理信号
        print(f"\n{'信号对比':<40} {'真实年化%':<10} {'代理年化%':<10} {'真实Sharpe':<12} {'代理Sharpe':<12}")
        print('-'*84)
        for sname,label in [("TB_real","TB"),("OSR_real","OSR"),("Chip_real","Chip")]:
            for pname,plabel in [("无择时","无择时"),("VolTiming","Vol"),("TrendTiming","Trend"),("MarketRegime","MR")]:
                for fname,fd in [("D3","D3"),("D5","D5"),("D10","D10")]:
                    real_name=f"{sname}_{pname}_{fname}"
                    proxy_name=f"{label}_{plabel}_{fname}"
                    real_r=next((r for r in step4_results if r['name']==real_name),None)
                    proxy_r=v8_map.get(proxy_name)
                    if real_r and proxy_r:
                        print(f"{sname}+{pname}+{fname:<12} {real_r['annual_return']*100:>8.2f}% {proxy_r['annual_return']*100:>8.2f}% {real_r['sharpe']:>10.3f} {proxy_r['sharpe']:>10.3f}")

    # Type E最佳结果
    logger.info("\n=== Type E最佳结果 ===")
    best=sorted(type_e_results,key=lambda x:x['sharpe'],reverse=True)[:5]
    for r in best:
        logger.info(f"  {r['name']}: 年化={r['annual_return']*100:.2f}% Sharpe={r['sharpe']:.3f} 回撤={r['max_drawdown']*100:.2f}% Calmar={r['calmar']:.3f}")

    logger.info(f"\n结果已保存至: {os.path.join(X3_DIR,'results.json')}")
    return results

if __name__=="__main__":
    main()
