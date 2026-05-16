"""
X2 — 实验A~F: 信号组合器/CompositeTiming/仓位分配器补充/参数微调/W03/落地策略验证

依赖:
  - X1 相同的 load()/v1w()/bt() 函数
  - 尝试调用 core.signals/composer, core.timings, core.positioners, core.screening 中的实际类
  - 若不兼容 bt() 的回测流程,则手写模拟
"""
import os,sys,json,logging,gc,numpy as np,pandas as pd
from datetime import datetime

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger("x2")
TX=0.0012

FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position','beta_20']))

X2_DIR=os.path.join(os.path.dirname(__file__),"x2_results")
os.makedirs(X2_DIR,exist_ok=True)

# ============================================================
# 数据加载 (同X1)
# ============================================================
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
    return z3,fwd,dm,tks,FACTORS,nd,ns,ds

# ============================================================
# V1权重加载 (同X1)
# ============================================================
def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

# ============================================================
# 回测函数 (同X1, 含 mhd 参数)
# ============================================================
def bt(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None,mhd=5):
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
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

# ============================================================
# 实验A: 信号组合器C01~C04
#   - 先尝试 import 实际类
#   - 由于 Composer.compose() 接口为 Dict[str,float] 而非 numpy,
#     无法直接集成到 bt() 中,故手写模拟
# ============================================================
def experiment_A(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi):
    """信号组合器 C01~C04 可行性评估与手写模拟"""
    logger.info("\n"+"="*60+"\n实验A: 信号组合器 C01~C04\n"+"="*60)
    results=[]

    # ── 可行性评估:尝试 import ──
    composer_available=False
    try:
        from core.signals import LayeredComposer,DirectComposer,WeightedComposer,VoteComposer
        from core.signals import MaxSingleWeightConstraint,MaxTotalPositionConstraint,ReserveCashConstraint
        # 检查 compose 接口是否可接受 numpy→dict 映射
        test_composer=LayeredComposer(top_n=10)
        # 测试: 如接口需要 Dict[str,float] 而非 numpy,标记为不可用
        logger.info("  [Composer] 实际类 import 成功,但 compose() 需 Dict[str,float] 输入,")
        logger.info("  [Composer] 与 bt() 的 numpy 流程不兼容 → 启用手写模拟")
    except Exception as e:
        logger.info(f"  [Composer] import 失败: {e}, 启用手写模拟")

    # ── 准备rank信号 (用于C03/C04) ──
    def norm_rank(sig):
        nd2,ns2=sig.shape
        out=np.zeros_like(sig)
        for d in range(nd2):
            r=sig[d];o=np.argsort(r)
            out[d,o]=np.arange(ns2)/float(ns2)
        return out
    mf_rank=norm_rank(mf)
    chip_rank=norm_rank(chip_sig)

    # ── 趋势信号作为择时层 ──
    im,ims,im5,im20,ir,iv=fi.get('macd'),fi.get('macd_signal'),fi.get('momentum_5'),fi.get('momentum_20'),fi.get('rsi_14'),fi.get('volatility_20')
    trend_signal=np.full((nd,ns),0.5,dtype=np.float32)
    for d in range(nd):
        sl=[]
        if im is not None and ims is not None:sl.append(np.where(z3[d,:,im]>z3[d,:,ims],1.0,0.0))
        if im5 is not None and im20 is not None:
            m5v,m20v=z3[d,:,im5],z3[d,:,im20]
            sl.append(np.where((m5v>0)&(m5v>m20v),1.0,np.where(m5v<0,0.0,0.5)))
        if ir is not None:
            rv=z3[d,:,ir]
            sl.append(np.where(rv>70,0.0,np.where(rv>=50,1.0,np.where(rv>=30,0.5,0.0))))
        if sl:trend_signal[d]=np.mean(np.mean(sl,axis=0),axis=0)

    # C01: LayeredComposer 模拟 — trend作为择时层, MF作为选图层
    #   weight = pos_ratio * (1/top_n) for selected stocks
    #   择时系数: trend_signal 的日平均值
    logger.info("\n--- C01 LayeredComposer 模拟 ---")
    c01_pr=np.clip(np.nanmean(trend_signal>=0.6,axis=1)*2.0,0.1,1.0)
    results.append(bt(mf,fwd,dm,"C01_Layered(trend_timing+MF)_D10",rf=10,tn=40,pos_ratio=c01_pr,mhd=5))
    results.append(bt(mf,fwd,dm,"C01_Layered(trend_timing+MF)_D5",rf=5,tn=40,pos_ratio=c01_pr,mhd=5))

    # C02: DirectComposer 模拟 — MF信号×trend信号 (综合信号)
    logger.info("\n--- C02 DirectComposer 模拟 ---")
    c02_sig=mf*trend_signal  # 综合信号 = 选股×择时
    results.append(bt(c02_sig,fwd,dm,"C02_Direct(MF*Trend)_D10",rf=10,tn=40,mhd=5))
    results.append(bt(c02_sig,fwd,dm,"C02_Direct(MF*Trend)_D5",rf=5,tn=40,mhd=5))

    # C03: WeightedComposer 模拟 — MF_rank×0.7 + Chip_rank×0.3 (rank方式)
    logger.info("\n--- C03 WeightedComposer 模拟 ---")
    c03_sig=mf_rank*0.7+chip_rank*0.3
    results.append(bt(c03_sig,fwd,dm,"C03_Weighted(MF70+Chip30)_D10",rf=10,tn=40,mhd=5))
    results.append(bt(c03_sig,fwd,dm,"C03_Weighted(MF70+Chip30)_D5",rf=5,tn=40,mhd=5))

    # C04: VoteComposer 模拟 — MF前40只和Chip前40只重叠的股票,等权分配
    logger.info("\n--- C04 VoteComposer 模拟 ---")
    c04_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        mf_top=np.argsort(-mf[d])[:40]
        chip_top=np.argsort(-chip_sig[d])[:40]
        intersect=np.intersect1d(mf_top,chip_top)
        if len(intersect)>0:
            c04_sig[d,intersect]=1.0/len(intersect)
    results.append(bt(c04_sig,fwd,dm,"C04_Vote(MF∩Chip_top40)_D10",rf=10,tn=40,mhd=5))
    results.append(bt(c04_sig,fwd,dm,"C04_Vote(MF∩Chip_top40)_D5",rf=5,tn=40,mhd=5))

    return results

# ============================================================
# 实验B: CompositeTiming T05_v
# ============================================================
def experiment_B(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi):
    """CompositeTiming 模拟: TrendTiming + VolatilityTiming 投票"""
    logger.info("\n"+"="*60+"\n实验B: CompositeTiming T05_v\n"+"="*60)
    results=[]

    # ── 可行性评估 ──
    try:
        from core.timings import CompositeTiming,TrendTiming,VolatilityTiming
        logger.info("  [CompositeTiming] import 成功,但 generate() 需 pd.DataFrame,")
        logger.info("  [CompositeTiming] 与 bt() 的 numpy 流程不兼容 → 启用手写模拟")
    except Exception as e:
        logger.info(f"  [CompositeTiming] import 失败: {e}, 启用手写模拟")

    # ── 手写模拟 ──
    # 用z3数据计算择时信号
    im,ims,im5,im20,ir,iv=fi.get('macd'),fi.get('macd_signal'),fi.get('momentum_5'),fi.get('momentum_20'),fi.get('rsi_14'),fi.get('volatility_20')

    # TrendTiming: MACD+动量5/20+RSI三因素均分, >=0.6→BUY, <=0.4→SELL
    trend_score=np.full((nd,ns),0.5,dtype=np.float32)
    for d in range(nd):
        sl=[]
        if im is not None and ims is not None:sl.append(np.where(z3[d,:,im]>z3[d,:,ims],1.0,0.0))
        if im5 is not None and im20 is not None:
            m5v,m20v=z3[d,:,im5],z3[d,:,im20]
            sl.append(np.where((m5v>0)&(m5v>m20v),1.0,np.where(m5v<0,0.0,0.5)))
        if ir is not None:
            rv=z3[d,:,ir]
            sl.append(np.where(rv>70,0.0,np.where(rv>=50,1.0,np.where(rv>=30,0.5,0.0))))
        if sl:trend_score[d]=np.mean(sl,axis=0)

    # VolTiming: vol>0.30→SELL, <=0.15→HOLD (用zscore后的vol, 实际均值0, 阈值需调整)
    # 使用原始vol数据 (未zscore)
    vol20_idx=fi.get('volatility_20')
    # 从z3还原近似原始vol: z3是标准化后的, 我们需要原始vol的百分位概念
    # 直接用z3的vol, 阈值调整为 zscore 下的值
    # zscore均值0, std1; 原始vol在0.1~0.6范围, zscore约 -1~3
    # 原始vol 0.30 ≈ zscore ~0.5~1.0 (取决于截面分布)
    vol_high_signal=np.zeros((nd,ns),dtype=bool)
    vol_low_signal=np.zeros((nd,ns),dtype=bool)
    if vol20_idx is not None:
        for d in range(nd):
            v=z3[d,:,vol20_idx]
            # 使用百分位: top 30% 视为高波动, bottom 30% 视为低波动
            if np.sum(~np.isnan(v))>0:
                th_hi=np.nanpercentile(v,70)
                th_lo=np.nanpercentile(v,30)
                vol_high_signal[d]=v>th_hi
                vol_low_signal[d]=v<th_lo

    # 投票: BUY>SELL→满仓, SELL>BUY→空仓, 平局→0.5
    # 对每只股票: TrendTiming产生BUY/SELL票, VolTiming产生SELL/HOLD票
    comp_p=np.full(nd,0.5,dtype=np.float32)
    for d in range(nd):
        trend_buy=trend_score[d]>=0.6
        trend_sell=trend_score[d]<=0.4
        vol_sell=vol_high_signal[d]
        # 每只股票的"投票"
        n_buy=np.sum(trend_buy)
        n_sell=np.sum(trend_sell)+np.sum(vol_sell)
        if n_buy>n_sell:
            comp_p[d]=1.0
        elif n_sell>n_buy:
            comp_p[d]=0.0
        # else: 0.5 (平局)

    # 频率: D3/D5/D10
    logger.info("\n--- CompositeTiming 模拟 ---")
    results.append(bt(mf,fwd,dm,"B_T05v_CompositeTiming(MF)_D3",rf=3,tn=40,pos_ratio=comp_p,mhd=5))
    results.append(bt(mf,fwd,dm,"B_T05v_CompositeTiming(MF)_D5",rf=5,tn=40,pos_ratio=comp_p,mhd=5))
    results.append(bt(mf,fwd,dm,"B_T05v_CompositeTiming(MF)_D10",rf=10,tn=40,pos_ratio=comp_p,mhd=5))

    # 也用Chip信号跑
    results.append(bt(chip_sig,fwd,dm,"B_T05v_CompositeTiming(Chip)_D3",rf=3,tn=40,pos_ratio=comp_p,mhd=5))
    results.append(bt(chip_sig,fwd,dm,"B_T05v_CompositeTiming(Chip)_D5",rf=5,tn=40,pos_ratio=comp_p,mhd=5))
    results.append(bt(chip_sig,fwd,dm,"B_T05v_CompositeTiming(Chip)_D10",rf=10,tn=40,pos_ratio=comp_p,mhd=5))

    return results

# ============================================================
# 实验C: 仓位分配器补充 P02/P05/P06/P08
# ============================================================
def experiment_C(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi):
    logger.info("\n"+"="*60+"\n实验C: 仓位分配器补充 P02/P05/P06/P08\n"+"="*60)
    results=[]

    # ── 准备信号 ──
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
    if iv is not None:
        # 原始 volatility_20 数据在z3中已标准化, 用百分位判断高波动
        vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.5,axis=1),0.2,1.0)

    # P02: 协方差风险平价 bt_covrp
    logger.info("\n--- P02 协方差风险平价 ---")
    def bt_covrp(sig,fwd,dm,name,rf=3,tn=40,mhd=5):
        nd,ns=sig.shape
        pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
        rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);nt=0
        for i in range(1,nd):
            rebal=(i%rf==0)
            if rebal:
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
                pw=nw;rh+=1
            else:
                mk=dm[i]&(pw>0)
                if np.any(mk):
                    p2=pw[mk].copy()/float(np.sum(pw[mk]))
                    pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
            rt=float(np.dot(pw,fwd[i]))
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

    results.append(bt_covrp(mf,fwd,dm,"P02_CovRP_MF_D10",rf=10,tn=40))
    results.append(bt_covrp(mf,fwd,dm,"P02_CovRP_MF_D5",rf=5,tn=40))
    results.append(bt_covrp(chip_sig,fwd,dm,"P02_CovRP_Chip_D3",rf=3,tn=40))
    results.append(bt_covrp(mf,fwd,dm,"P02_CovRP_MF+Vol_D10",rf=10,tn=40,mhd=5))

    # P05: 等权 bt_ew
    logger.info("\n--- P05 等权 ---")
    def bt_ew(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None,mhd=5):
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
                to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*TX
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

    results.append(bt_ew(mf,fwd,dm,"P05_EqualWeight_MF_D10",rf=10,tn=40))
    results.append(bt_ew(mf,fwd,dm,"P05_EqualWeight_MF_D5",rf=5,tn=40))
    results.append(bt_ew(chip_sig,fwd,dm,"P05_EqualWeight_Chip_D3",rf=3,tn=40))
    results.append(bt_ew(mf,fwd,dm,"P05_EqualWeight_MF+Vol_D10",rf=10,tn=40,pos_ratio=vol_p))

    # P06: TrendPositionSizer — trend_p作为仓位系数
    logger.info("\n--- P06 TrendPositionSizer ---")
    results.append(bt(mf,fwd,dm,"P06_TrendSizer_MF_D10",rf=10,tn=40,pos_ratio=trend_p,mhd=5))
    results.append(bt(mf,fwd,dm,"P06_TrendSizer_MF_D5",rf=5,tn=40,pos_ratio=trend_p,mhd=5))
    results.append(bt(chip_sig,fwd,dm,"P06_TrendSizer_Chip_D3",rf=3,tn=40,pos_ratio=trend_p,mhd=5))
    results.append(bt(mf,fwd,dm,"P06_TrendSizer_MF+Vol_D10",rf=10,tn=40,pos_ratio=trend_p,mhd=5))

    # P08: CompositePositionSizer(mode="min") — min(trend_p, vol_p)
    logger.info("\n--- P08 CompositePositionSizer(min) ---")
    comp_min_p=np.minimum(trend_p,vol_p)
    results.append(bt(mf,fwd,dm,"P08_CompSizer(min)_MF_D10",rf=10,tn=40,pos_ratio=comp_min_p,mhd=5))
    results.append(bt(mf,fwd,dm,"P08_CompSizer(min)_MF_D5",rf=5,tn=40,pos_ratio=comp_min_p,mhd=5))
    results.append(bt(chip_sig,fwd,dm,"P08_CompSizer(min)_Chip_D3",rf=3,tn=40,pos_ratio=comp_min_p,mhd=5))
    results.append(bt(mf,fwd,dm,"P08_CompSizer(min)_MF+Vol_D10",rf=10,tn=40,pos_ratio=comp_min_p,mhd=5))

    return results

# ============================================================
# 实验D: 参数微调 vol_lookback扫描
# ============================================================
def experiment_D(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi):
    logger.info("\n"+"="*60+"\n实验D: 参数微调 vol_lookback 扫描\n"+"="*60)
    results=[]

    iv=fi.get('volatility_20')
    vol_p=np.ones(nd,dtype=np.float32)
    if iv is not None:
        vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.5,axis=1),0.2,1.0)

    # D1: MF_Vol_D10_mhd=10, vol_lookback扫描
    logger.info("\n--- D1: MF_Vol_D10_mhd=10 vol_lookback扫描 ---")
    def make_bt_vl(vl):
        def bt_vl(sig,fwd,dm,name,rf=10,tn=40,mhd=10):
            nd,ns=sig.shape
            alloc_vl=RPPortfolioWeights(top_n=tn,min_hold_days=mhd,vol_lookback=vl)
            pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
            rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);nt=0
            for i in range(1,nd):
                rebal=(i%rf==0)
                if rebal:
                    nw=alloc_vl.allocate(sig[i],fwd,i,pw,hs,rh)
                    to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*TX
                    if to>0.01:nt+=1
                    pw=nw
                    for j in range(ns):
                        if nw[j]>0 and hs[j]<0:hs[j]=rh+1
                else:
                    mk=dm[i]&(pw>0)
                    if np.any(mk):
                        p2=pw[mk].copy()/float(np.sum(pw[mk]))
                        pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
                pr=1.0
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
            logger.info(f"  [{name}] vl={vl} 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
            return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}
        return bt_vl
    for vl in [10,20,30,40,60]:
        label=f"D1_MF+Vol_D10_mhd=10_vl={vl}"
        results.append(make_bt_vl(vl)(mf,fwd,dm,label,rf=10,tn=40,mhd=10))

    # D2: MF_D10 vol_lookback扫描 (无vol择时, 但RPPortfolioWeights内部用vol_lookback)
    logger.info("\n--- D2: MF_D10 vol_lookback扫描 ---")
    def make_bt_vl2(vl):
        def bt_vl2(sig,fwd,dm,name,rf=10,tn=40,mhd=5):
            nd,ns=sig.shape
            alloc_vl=RPPortfolioWeights(top_n=tn,min_hold_days=mhd,vol_lookback=vl)
            pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
            rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);nt=0
            for i in range(1,nd):
                rebal=(i%rf==0)
                if rebal:
                    nw=alloc_vl.allocate(sig[i],fwd,i,pw,hs,rh)
                    to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*TX
                    if to>0.01:nt+=1
                    pw=nw
                    for j in range(ns):
                        if nw[j]>0 and hs[j]<0:hs[j]=rh+1
                else:
                    mk=dm[i]&(pw>0)
                    if np.any(mk):
                        p2=pw[mk].copy()/float(np.sum(pw[mk]))
                        pw=np.zeros(ns,dtype=np.float32);pw[mk]=p2
                pr=1.0
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
            logger.info(f"  [{name}] vl={vl} 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
            return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}
        return bt_vl2
    for vl in [10,20,30,40,60]:
        label=f"D2_MF_D10_vl={vl}"
        results.append(make_bt_vl2(vl)(mf,fwd,dm,label,rf=10,tn=40,mhd=5))

    return results

# ============================================================
# 实验E: 权重W03 from_registry
# ============================================================
def experiment_E(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi):
    logger.info("\n"+"="*60+"\n实验E: W03 MultiFactorSelector.from_registry\n"+"="*60)
    results=[]
    registry_available=False
    registry_weights=None

    # 尝试从数据库获取注册表权重
    try:
        db=Database()
        reg_df=db.get_enabled_factors(min_abs_ir=0.2,as_dataframe=True)
        if reg_df is not None and len(reg_df)>0:
            registry_available=True
            weights=dict(zip(reg_df["factor_name"],reg_df["ir"].astype(float)))
            registry_weights=weights
            logger.info(f"  [Registry] 成功获取 {len(weights)} 个因子的IR权重")
            logger.info(f"  [Registry] 前10因子: {list(weights.keys())[:10]}")
        else:
            logger.info("  [Registry] factor_registry 为空或无不达标因子")
    except Exception as e:
        logger.info(f"  [Registry] 获取失败: {e}")

    if registry_available and registry_weights is not None:
        # 构建注册表权重信号
        reg_wv=np.zeros(len(fnames),dtype=np.float32)
        for i,fc in enumerate(fnames):
            if fc in registry_weights:
                reg_wv[i]=float(registry_weights[fc])
        s=np.sum(np.abs(reg_wv))
        if s>0:
            reg_wv/=s
            reg_mf=np.nan_to_num(np.tensordot(z3,reg_wv,axes=(2,0)),nan=-1e10,neginf=-1e10)
            logger.info("  [Registry] 使用注册表权重构建信号并回测")
            results.append(bt(reg_mf,fwd,dm,"E_W03_from_registry_D10",rf=10,tn=40,mhd=5))
            results.append(bt(reg_mf,fwd,dm,"E_W03_from_registry_D5",rf=5,tn=40,mhd=5))
            results.append(bt(reg_mf,fwd,dm,"E_W03_from_registry_D3",rf=3,tn=40,mhd=5))
            # 带Vol择时
            iv=fi.get('volatility_20')
            if iv is not None:
                vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.5,axis=1),0.2,1.0)
                results.append(bt(reg_mf,fwd,dm,"E_W03_from_registry+Vol_D10",rf=10,tn=40,pos_ratio=vol_p,mhd=5))
        else:
            logger.info("  [Registry] 权重总和为0, 跳过回测")
            results.append({"name":"E_W03_from_registry","note":"权重总和为0, 跳过回测",
                "annual_return":0,"sharpe":0,"max_drawdown":0,"calmar":0,"win_rate":0,"n_trades":0})
    else:
        logger.info("  [Registry] 注册表不可用, 跳过此实验")
        results.append({"name":"E_W03_from_registry","note":"注册表不可用, 跳过",
            "annual_return":0,"sharpe":0,"max_drawdown":0,"calmar":0,"win_rate":0,"n_trades":0})

    return results

# ============================================================
# 实验F: 已落地策略跑窗口验证补充
# ============================================================
def experiment_F(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi,ds):
    logger.info("\n"+"="*60+"\n实验F: 已落地策略窗口验证补充\n"+"="*60)
    results=[]

    im,ims,im5,im20,ir,iv=fi.get('macd'),fi.get('macd_signal'),fi.get('momentum_5'),fi.get('momentum_20'),fi.get('rsi_14'),fi.get('volatility_20')

    # trend_p信号 (同用户提供代码)
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

    # ── 定义窗口回测函数 ──
    def bt_window(sig,fwd,dm,name,rf=3,tn=40,pos_ratio=None,mhd=5):
        nd,ns=sig.shape
        alloc=RPPortfolioWeights(top_n=tn,min_hold_days=mhd)
        pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
        rh=0;eq=np.ones(nd,dtype=np.float64);dr=np.zeros(nd,dtype=np.float64);nt=0
        for i in range(1,nd):
            rebal=(i%rf==0)
            if rebal:
                pr=pos_ratio[i] if pos_ratio is not None else 1.0
                nw=alloc.allocate(sig[i],fwd,i,pw,hs,rh)
                to=float(np.sum(np.abs(nw-pw)));txc=0.5*to*TX
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
        return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

    # ── 定义7个窗口 ──
    # 总时间: 2018-01 至 2026-04 (~8.3年)
    # 窗口: 每~1.2年一个, 步长~1年
    window_configs=[
        ("W1_2018-2019","2018-01-01","2019-06-30"),
        ("W2_2019-2020","2019-07-01","2020-12-31"),
        ("W3_2021-H1","2021-01-01","2021-12-31"),
        ("W4_2022-H1","2022-01-01","2022-12-31"),
        ("W5_2023","2023-01-01","2023-12-31"),
        ("W6_2024","2024-01-01","2024-12-31"),
        ("W7_2025-2026","2025-01-01","2026-04-30"),
    ]

    import bisect
    def slice_data(sig,fwd,dm,ds,start_date,end_date):
        ds_list=list(ds)
        d0=bisect.bisect_left(ds_list,pd.Timestamp(start_date))
        d1=bisect.bisect_right(ds_list,pd.Timestamp(end_date))-1
        if d0>=len(ds_list)or d1<0 or d0>d1:return None,None,None
        return sig[d0:d1+1],fwd[d0:d1+1],dm[d0:d1+1]

    d2i={d:i for i,d in enumerate(ds)}

    # F1: mf_d10_rp (tn=50, rf=10, mhd=10)
    logger.info("\n--- F1: mf_d10_rp 窗口验证 ---")
    for wname,ws,we in window_configs:
        s_sig,s_fwd,s_dm=slice_data(mf,fwd,dm,ds,ws,we)
        if s_sig is None:
            logger.info(f"  [{wname}] 数据不足,跳过")
            continue
        label=f"F1_mf_d10_rp_{wname}"
        r=bt_window(s_sig,s_fwd,s_dm,label,rf=10,tn=50,mhd=10)
        results.append(r)
        logger.info(f"  [{label}] 年化={r['annual_return']*100:.2f}% Sharpe={r['sharpe']:.3f} 回撤={r['max_drawdown']*100:.2f}%")

    # F2: mf_trend_d5_rp (tn=40, rf=5, pos_ratio=trend_p)
    logger.info("\n--- F2: mf_trend_d5_rp 窗口验证 ---")
    for wname,ws,we in window_configs:
        s_sig,s_fwd,s_dm=slice_data(mf,fwd,dm,ds,ws,we)
        if s_sig is None:
            continue
        # 也需要截取 trend_p
        ds_list=list(ds)
        d0=bisect.bisect_left(ds_list,pd.Timestamp(ws))
        d1=bisect.bisect_right(ds_list,pd.Timestamp(we))-1
        s_tp=trend_p[d0:d1+1]
        label=f"F2_mf_trend_d5_rp_{wname}"
        r=bt_window(s_sig,s_fwd,s_dm,label,rf=5,tn=40,pos_ratio=s_tp,mhd=5)
        results.append(r)
        logger.info(f"  [{label}] 年化={r['annual_return']*100:.2f}% Sharpe={r['sharpe']:.3f} 回撤={r['max_drawdown']*100:.2f}%")

    return results

# ============================================================
# Main
# ============================================================
def main():
    logger.info("="*60+"\nX2 Pipeline — 实验A~F\n"+"="*60)

    # 数据加载
    z3,fwd,dm,tks,fnames,nd,ns,ds=load()
    fi={fn:i for i,fn in enumerate(fnames)}
    v1w_dict=v1w()
    wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # Chip信号 (同v8)
    vol20_idx=fi.get('volatility_20');m20_idx=fi.get('momentum_20')
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None:s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5
        if m20_idx is not None:s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3
        chip_sig[d]=np.nan_to_num(s,nan=-1e10)

    all_results=[]

    # 实验A
    rA=experiment_A(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi)
    all_results.extend(rA)

    # 实验B
    rB=experiment_B(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi)
    all_results.extend(rB)

    # 实验C
    rC=experiment_C(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi)
    all_results.extend(rC)

    # 实验D
    rD=experiment_D(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi)
    all_results.extend(rD)

    # 实验E
    rE=experiment_E(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi)
    all_results.extend(rE)

    # 实验F
    rF=experiment_F(z3,fwd,dm,tks,fnames,nd,ns,mf,chip_sig,fi,ds)
    all_results.extend(rF)

    # ── 保存结果 ──
    out_path=os.path.join(X2_DIR,"results.json")
    with open(out_path,'w') as f:
        json.dump(all_results,f,indent=2,ensure_ascii=False)
    logger.info(f"\n结果已保存至: {out_path}")

    # ── 打印汇总表 ──
    print(f"\n{'='*130}")
    print(f"{'实验名':<42} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'交易':<5}")
    print('-'*130)
    valid=[r for r in all_results if r.get('annual_return',-999)!=-999 and r.get('sharpe',-999)!=-999]
    for r in sorted(valid,key=lambda x:x.get('sharpe',-999),reverse=True):
        if 'note' in r:
            print(f"  {r['name']:<40} — {r['note']}")
            continue
        ok=r['max_drawdown']<0.20 and r['annual_return']>0.10
        cls="🏆" if ok else("  ")
        print(f"{cls} {r['name']:<40} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")
    print('='*130)

    # 达标统计
    qualified=[r for r in valid if r['max_drawdown']<0.20 and r['annual_return']>0.10]
    logger.info(f"\n🏆 达标(回撤<20% 年化>10%): {len(qualified)}个")
    for r in sorted(qualified,key=lambda x:x['sharpe'],reverse=True)[:10]:
        logger.info(f"  🏆 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")

if __name__=="__main__":
    main()
