"""
V9c — 多策略组合实验

将低相关性的策略加权组合，找出"回撤<20% 且 年化>20%"的组合。
"""
import os,sys,json,logging,gc,numpy as np,pandas as pd
from datetime import datetime

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger("v9c")
TX=0.0012
FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position']))
V9C_DIR=os.path.join(os.path.dirname(__file__),"v9c_combo");os.makedirs(V9C_DIR,exist_ok=True)

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
    return z3,fwd,dm,tks,FACTORS,nd,ns

def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

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
    z3,fwd,dm,tks,fnames,nd,ns=load()
    fi={fn:i for i,fn in enumerate(fnames)}
    v1w_dict=v1w()
    wv=np.zeros(len(fnames),dtype=np.float32)
    for fi_i,fc in enumerate(fnames):
        if fc in v1w_dict:wv[fi_i]=float(v1w_dict[fc])
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    mf=np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

    # Chip信号
    vol20_idx=fi.get('volatility_20');m20_idx=fi.get('momentum_20');iv=fi.get('volatility_20')
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None:s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5
        if m20_idx is not None:s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3
        chip_sig[d]=np.nan_to_num(s,nan=-1e10)

    # 预计算各策略仓位系数
    # MF_Vol_D10_mhd=10: 最佳微调参数
    vol_p=np.ones(nd,dtype=np.float32)
    if iv is not None:vol_p=np.clip(1.0-np.mean(z3[:,:,iv]>0.30,axis=1),0.2,1.0)

    res=[]
    logger.info("="*60+"\n多策略组合实验\n"+"="*60)

    # ── 各策略单独回测（基线） ──
    logger.info("\n各策略基线:")
    res.append(bt(mf,fwd,dm,"MF_D10_mhd=10(基线)",rf=10,tn=50,mhd=10))
    res.append(bt(mf,fwd,dm,"MF+Vol_D10_mhd=10(基线)",rf=10,tn=50,pos_ratio=vol_p,mhd=10))
    res.append(bt(chip_sig,fwd,dm,"Chip_D3(基线)",rf=3,tn=40))

    # ── MF × Chip 加权组合 ──
    # 组合信号 = w1*mf + w2*chip (截面标准化后合并)
    logger.info("\nMF × Chip 加权组合:")
    def norm_rank(sig):
        nd2,ns2=sig.shape
        out=np.zeros_like(sig)
        for d in range(nd2):
            r=sig[d];o=np.argsort(r)
            out[d,o]=np.arange(ns2)/float(ns2)
        return out
    mf_rank=norm_rank(mf);chip_rank=norm_rank(chip_sig)

    for w_mf in [0.3,0.4,0.5,0.6,0.7,0.8]:
        w_chip=1.0-w_mf
        combo=mf_rank*w_mf+chip_rank*w_chip
        label=f"MF{w_mf:.0f}+Chip{w_chip:.0f}_D10"
        res.append(bt(combo,fwd,dm,label,rf=10,tn=40,mhd=5))

    # 带择时的组合
    for w_mf in [0.5,0.7]:
        w_chip=1.0-w_mf
        combo=mf_rank*w_mf+chip_rank*w_chip
        label=f"MF{w_mf:.0f}+Chip{w_chip:.0f}+Vol_D10"
        res.append(bt(combo,fwd,dm,label,rf=10,tn=40,pos_ratio=vol_p,mhd=10))

    # ── MF × Chip × RSI14 三路组合 ──
    logger.info("\nMF × Chip × RSI14 三路组合:")
    rsi_sig=np.nan_to_num(z3[:,:,fi['rsi_14']],nan=-1e10,neginf=-1e10) if fi.get('rsi_14') is not None else mf
    rsi_rank=norm_rank(rsi_sig)

    for weights in [(0.5,0.3,0.2),(0.4,0.4,0.2),(0.6,0.2,0.2),(0.3,0.3,0.4)]:
        combo=mf_rank*weights[0]+chip_rank*weights[1]+rsi_rank*weights[2]
        label=f"MF{weights[0]:.0f}+Chip{weights[1]:.0f}+RSI{weights[2]:.0f}_D10"
        res.append(bt(combo,fwd,dm,label,rf=10,tn=40,mhd=5))
        # 带择时的版本
        res.append(bt(combo,fwd,dm,label+"+Vol",rf=10,tn=40,pos_ratio=vol_p,mhd=10))

    # ── 频率交叉: 最好的3个组合换频率 ──
    logger.info("\n最佳组合 × 频率扫描:")
    combo50=mf_rank*0.5+chip_rank*0.5
    for fd in [5,7,8,10,12]:
        res.append(bt(combo50,fwd,dm,f"MF50+Chip50_D{fd}",rf=fd,tn=40,mhd=5))

    # 保存
    with open(os.path.join(V9C_DIR,"results.json"),'w') as f:
        json.dump(res,f,indent=2,ensure_ascii=False)

    # 输出
    qualified=[r for r in res if r['max_drawdown']<0.20 and r['annual_return']>0.10]
    near=[r for r in res if 0.20<=abs(r['max_drawdown'])<0.25 and r['annual_return']>0.18]

    print(f"\n{'='*120}")
    print(f"{'实验':<40} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'换手':<5}")
    print('-'*120)
    for r in sorted(res,key=lambda x:x['sharpe'],reverse=True):
        ok=r['max_drawdown']<0.20 and r['annual_return']>0.10
        cls="🏆" if ok else("🔥" if 0.20<=abs(r['max_drawdown'])<0.25 and r['annual_return']>0.18 else("  "))
        print(f"{cls} {r['name']:<38} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")
    print('='*120)

    logger.info(f"\n🏆 达标(回撤<20% 年化>10%): {len(qualified)}个")
    for r in sorted(qualified,key=lambda x:x['sharpe'],reverse=True)[:10]:
        logger.info(f"  🏆 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")
    logger.info(f"\n🔥 接近(回撤20~25% 年化>18%): {len(near)}个")
    for r in sorted(near,key=lambda x:x['sharpe'],reverse=True)[:10]:
        logger.info(f"  🔥 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")

if __name__=="__main__":
    main()
