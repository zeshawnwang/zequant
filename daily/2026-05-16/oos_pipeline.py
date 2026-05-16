"""
OOS验证 — 全部19策略的训练/测试拆分验证。

训练期: 2019-01-02 ~ 2024-06-28
测试期: 2024-07-01 ~ 2026-04-30

训练期用来确认参数，测试期是out-of-sample，不准改参数。
"""
import os,sys,json,logging,gc,numpy as np,pandas as pd
from datetime import datetime

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger("oos")
TX=0.0012

FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5','momentum_20',
    'volume_ratio','boll_position','beta_20']))
# 新因子列
NEW_FACTORS=['ma5','ma10','ma20','ma21','ma60','ma120','ma_alignment_score','ma60_trend',
    'ma120_trend','macd_above_zero','macd_golden_cross','volume_breakout_ratio',
    'volume_contraction','ma_convergence','chip_concentration','ma_angle_20']
ALL_FACTORS=list(set(FACTORS+NEW_FACTORS))

OOS_DIR=os.path.join(os.path.dirname(__file__),"oos_results");os.makedirs(OOS_DIR,exist_ok=True)

TRAIN_END="2024-06-28"

def load():
    db=Database()
    all_cols=db.list_factor_columns()
    available=[c for c in ALL_FACTORS if c in all_cols]
    logger.info(f"可用因子: {len(available)}")
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

def load_ga_weights():
    p1="daily/2026-05-16/x4_x5_results/x5_results.json"
    p2="daily/2026-05-13/v1/decoupled_results.json"
    ga_w=None
    for p in [p1,p2]:
        fp=os.path.join(os.path.dirname(__file__),'..','..',p)
        if os.path.exists(fp):
            with open(fp) as f:
                data=json.load(f)
                if isinstance(data,list) and len(data)>0:
                    if 'weights' in data[0]:
                        ga_w={k:float(v) for k,v in data[0]['weights'].items()}
                        break
                    elif 'configs' in data[0]:
                        ga_w={k:float(v) for k,v in data[0]['configs'][0]['weights'].items()}
                        break
    return ga_w or {}

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
    tr=float(eq[-1]/eq[0]-1.0);ny=nd/252.0
    ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
    lr=np.log(eq[1:]/eq[:-1])
    sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
    cal=ar/abs(mdd)if abs(mdd)>0 else 0
    wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),"max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":nt}

def bt_series(sig,fwd,dm,rf=3,tn=40,pos_ratio=None,mhd=5):
    """返回每日收益序列。"""
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
        dr[i]=0.0 if(np.isnan(rt)or np.isinf(rt))else rt;rh+=1
    return dr

def main():
    z3,fwd,dm,tks,fnames,nd,ns,ds=load()
    fi={fn:i for i,fn in enumerate(fnames)}
    v1w_dict=v1w()
    ga_dict=load_ga_weights()
    res=[]

    # 找到train/test分界点
    train_end_dt=pd.Timestamp(TRAIN_END)
    split_idx=next(i for i,d in enumerate(ds) if d>train_end_dt)
    logger.info(f"训练集: {len(ds[:split_idx])}天 ({ds[0]}~{ds[split_idx-1]})")
    logger.info(f"测试集: {len(ds[split_idx:])}天 ({ds[split_idx]}~{ds[-1]})")

    # ── 构建所有信号 ──
    wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # GA权重
    gaw=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in ga_dict:gaw[fi_i]=float(ga_dict[fc])
    s2=np.sum(np.abs(gaw));gaw/=s2 if s2>0 else 1
    ga=np.nan_to_num(np.tensordot(z3,gaw,axes=(2,0)),nan=-1e10,neginf=-1e10)

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
        if returns_idx:='returns' in fi:s_osr+=np.where(z3[d,:,fi['returns']]<-0.5,1.0,0.0)*0.3
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

    # ── 策略定义 ──
    strategies=[
        ("v1_ga_rp",     mf,          3, 40, None,   5),
        ("v4_mf_rp",     mf,          3, 40, None,   5),
        ("v4_mf_tv_rp",  mf,          3, 40, vol_p,  5),
        ("mf_d10_rp",    mf,         10, 50, None,  10),
        ("mf_vol_d10_rp",mf,         10, 50, vol_p, 10),
        ("mf_trend_d5_rp",mf,         5, 40, trend_p,5),
        ("chip_rp",      chip_sig,    3, 40, None,   5),
        ("chip_vol_rp",  chip_sig,    3, 40, vol_p,  5),
        ("chip_covrp",   chip_sig,    3, 40, None,   5),
        ("chip_equal_d3",chip_sig,    3, 40, None,   5),
        ("osr_d10",      osr_sig,    10, 40, None,   5),
        ("osr_vol_eq_d10",osr_sig,   10, 40, vol_p,  5),
        ("ga_d10",       ga,         10, 40, None,   5),
        ("ga_d5",        ga,          5, 40, None,   5),
    ]

    # 组合策略（资金分配法）
    combo_strategies=[
        ("mf60_chip40_combo",   mf, chip_sig,   None, 0.6, 0.4, 10, 3, 50, 40, None, None, 10),
        ("mf50_chip50_combo",   mf, chip_sig,   None, 0.5, 0.5, 10, 3, 50, 40, None, None, 10),
        ("mf50_chipcovrp50_combo",mf, chip_sig, None, 0.5, 0.5, 10, 3, 50, 40, None, None, 10),
        ("c01_layered_d5",      mf, chip_sig,   None, 0.0, 0.0,  5, 3, 40, 40, trend_p, None, 10),
        ("ga_covrp_combo",      ga,  chip_sig,   None, 0.6, 0.4, 10, 3, 40, 40, None, None, 10),
    ]

    logger.info("="*60)
    logger.info(f"OOS验证 — {len(strategies)+len(combo_strategies)}策略")
    logger.info("="*60)

    # ── 单策略测试 ──
    for sname, sig, rf, tn, pr, mhd in strategies:
        logger.info(f"\n--- {sname} ---")
        # 训练集
        sig_train=sig[:split_idx]
        fwd_train=fwd[:split_idx]
        dm_train=dm[:split_idx]
        r1=bt(sig_train,fwd_train,dm_train,f"{sname}_train",rf=rf,tn=tn,pos_ratio=pr,mhd=mhd)

        # 测试集
        sig_test=sig[split_idx:]
        fwd_test=fwd[split_idx:]
        dm_test=dm[split_idx:]
        r2=bt(sig_test,fwd_test,dm_test,f"{sname}_test",rf=rf,tn=tn,pos_ratio=pr,mhd=mhd)
        r2["name"]=sname
        res.append(r2)

    # ── 组合策略 ──
    for sname, sig1, sig2, sig3, w1, w2, rf1, rf2, tn1, tn2, pr, pr2, mhd in combo_strategies:
        logger.info(f"\n--- {sname} (组合) ---")
        if w1==0.0 and w2==0.0:  # c01_layered: 用择时过滤+MF
            dr_train=bt_series(sig1[:split_idx]*trend_p[:split_idx, np.newaxis],fwd[:split_idx],dm[:split_idx],
                               rf=rf1,tn=tn1,pos_ratio=pr if pr is not None else trend_p[:split_idx],mhd=mhd)
            dr_test=bt_series(sig1[split_idx:]*trend_p[split_idx:, np.newaxis],fwd[split_idx:],dm[split_idx:],
                              rf=rf1,tn=tn1,pos_ratio=pr if pr is not None else trend_p[split_idx:],mhd=mhd)
        else:
            dr_a_train=bt_series(sig1[:split_idx],fwd[:split_idx],dm[:split_idx],rf=rf1,tn=tn1,pos_ratio=pr,mhd=mhd)
            dr_b_train=bt_series(sig2[:split_idx],fwd[:split_idx],dm[:split_idx],rf=rf2,tn=tn2,pos_ratio=pr2,mhd=mhd)
            dr_train=dr_a_train*w1+dr_b_train*w2
            dr_a_test=bt_series(sig1[split_idx:],fwd[split_idx:],dm[split_idx:],rf=rf1,tn=tn1,pos_ratio=pr,mhd=mhd)
            dr_b_test=bt_series(sig2[split_idx:],fwd[split_idx:],dm[split_idx:],rf=rf2,tn=tn2,pos_ratio=pr2,mhd=mhd)
            dr_test=dr_a_test*w1+dr_b_test*w2

        def eval_from_dr(dr,name):
            nd2=len(dr);eq=np.ones(nd2)
            for i in range(1,nd2):eq[i]=eq[i-1]*(1.0+dr[i])
            tr=float(eq[-1]/eq[0]-1.0);ny=nd2/252.0
            ar=(float(eq[-1]/eq[0]))**(1.0/max(ny,0.5))-1.0
            lr=np.log(eq[1:]/eq[:-1])
            sp=float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
            cm=np.maximum.accumulate(eq);dd=(eq-cm)/cm;mdd=float(np.min(dd))
            cal=ar/abs(mdd)if abs(mdd)>0 else 0
            wr=int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
            logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
            return{"name":name,"annual_return":round(ar,4),"sharpe":round(sp,4),
                   "max_drawdown":round(mdd,4),"calmar":round(cal,4),"win_rate":round(wr,4),"n_trades":0}

        eval_from_dr(dr_train,f"{sname}_train")
        r2=eval_from_dr(dr_test,f"{sname}_test")
        r2["name"]=sname
        res.append(r2)

    # ── 保存 ──
    with open(os.path.join(OOS_DIR,"results.json"),'w') as f:
        json.dump(res,f,indent=2,ensure_ascii=False)

    # ── 输出排名 ──
    print(f"\n{'='*90}")
    print(f"{'测试期(OOS)排名':^90}")
    print(f"{'策略':<28} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}")
    print('-'*90)
    for r in sorted(res,key=lambda x:x['sharpe'],reverse=True):
        print(f"  {r['name']:<26} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f}")
    print('='*90)

    # 标注达标
    qualified=[r for r in res if abs(r['max_drawdown'])<0.20 and r['annual_return']>0.05]
    logger.info(f"\n测试期达标(回撤<20% 年化>5%): {len(qualified)}个")
    for r in sorted(qualified,key=lambda x:x['sharpe'],reverse=True):
        logger.info(f"  🏆 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")

if __name__=="__main__":
    main()
