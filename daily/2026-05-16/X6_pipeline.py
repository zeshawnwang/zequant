"""
X6 — 多策略资金分配优化

用bt_series + combo_from_series方法，对5个子策略做组合优化：
  - 子策略池：MF_D10, MF+Vol_D10, Chip_D3, Chip_Vol_D3, Chip_CovRP_D3
  - 二路组合：每对策略，w从0.1到0.9步长0.1
  - 三路组合：最优3个，w总和=1.0步长0.2
  - 筛选：Sharpe最高 / Calmar最高 / 回撤<20%且年化>10%
"""
import os,sys,json,logging,numpy as np,pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)]);logger=logging.getLogger("X6")
TX=0.0012

FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99','returns','rsi_14','volatility_20','macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position']))

RESULTS_DIR=os.path.join(os.path.dirname(__file__),"x6_combo_opt");os.makedirs(RESULTS_DIR,exist_ok=True)

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

def bt_series_covrp(sig,fwd,dm,rf=3,tn=40,pos_ratio=None,tx=TX):
    nd,ns=sig.shape
    pw=np.zeros(ns,dtype=np.float32);hs=np.full(ns,-1,dtype=np.int32)
    rh=0;dr=np.zeros(nd,dtype=np.float64)
    for i in range(1,nd):
        rebal=(i%rf==0);txc=0.0
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

def main():
    logger.info("="*60+"\nX6 — 多策略资金分配优化\n"+"="*60)
    z3,fwd,dm,tks,fnames,nd,ns,ds=load()
    fi={fn:i for i,fn in enumerate(fnames)}

    # ── MF信号 (V1权重加权) ──
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

    # ── Chip信号 (低波动+低动量，同V8) ──
    chip_sig=np.full((nd,ns),-np.inf,dtype=np.float32)
    vol20_idx=fi.get('volatility_20');m20_idx=fi.get('momentum_20')
    for d in range(nd):
        s=np.zeros(ns)
        if vol20_idx is not None:s+=np.where(z3[d,:,vol20_idx]<-0.3,1.0,0.0)*0.5
        if m20_idx is not None:s+=np.where(np.abs(z3[d,:,m20_idx])<0.3,1.0,0.0)*0.3
        chip_sig[d]=np.nan_to_num(s,nan=-1e10)

    # ── 5个子策略收益序列 ──
    logger.info("生成5个子策略收益序列...")
    strategies={
        "MF_D10":          bt_series_with_tx(mf,fwd,dm,rf=10,tn=50,mhd=10),
        "MF+Vol_D10":      bt_series_with_tx(mf,fwd,dm,rf=10,tn=50,mhd=10,pos_ratio=vol_p),
        "Chip_D3":         bt_series_with_tx(chip_sig,fwd,dm,rf=3,tn=40,mhd=5),
        "Chip_Vol_D3":     bt_series_with_tx(chip_sig,fwd,dm,rf=3,tn=40,mhd=5,pos_ratio=vol_p),
        "Chip_CovRP_D3":   bt_series_covrp(chip_sig,fwd,dm,rf=3,tn=40),
    }

    # ── 单独评估 ──
    solo_results={}
    for sname,sseries in strategies.items():
        r=combo_from_series(sseries,np.zeros(nd),w1=1.0,w2=0.0)
        r["name"]=sname
        solo_results[sname]=r
        logger.info(f"  {sname}: Sharpe={r['sharpe']:.3f} 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}%")

    # ── 二路组合 ──
    s_names=list(strategies.keys())
    all_combos_2way=[]
    logger.info("\n=== 二路组合 ===")
    for i in range(len(s_names)):
        for j in range(i+1,len(s_names)):
            na,nb=s_names[i],s_names[j]
            da=strategies[na];db=strategies[nb]
            for w1 in np.arange(0.1,1.0,0.1):
                w1r=round(w1,1);w2r=round(1.0-w1,1)
                r=combo_from_series(da,db,w1=w1r,w2=w2r)
                r["name"]=f"{na}({w1r})×{nb}({w2r})"
                r["type"]="2way";r["s1"]=na;r["s2"]=nb;r["w1"]=w1r;r["w2"]=w2r
                all_combos_2way.append(r)

    # ── 三路组合：选Sharpe最好的3个策略 ──
    ranked=sorted(solo_results.items(),key=lambda x:x[1]["sharpe"],reverse=True)
    top3_names=[ranked[0][0],ranked[1][0],ranked[2][0]]
    logger.info(f"\n三路组合使用Top3策略: {top3_names}")
    d1,d2,d3=strategies[top3_names[0]],strategies[top3_names[1]],strategies[top3_names[2]]

    all_combos_3way=[]
    logger.info("\n=== 三路组合 ===")
    w3_values=np.arange(0.0,1.01,0.2)
    for w1 in w3_values:
        for w2 in w3_values:
            w3r=round(1.0-w1-w2,1)
            if w3r<0 or w3r>1.0:continue
            if round(w1+w2+w3r,1)!=1.0:continue
            w1r=round(w1,1);w2r=round(w2,1)
            if w1r==0 or w2r==0 or w3r==0:continue
            r=combo_from_series(d1,d2,d3,w1=w1r,w2=w2r,w3=w3r)
            r["name"]=f"{top3_names[0]}({w1r})×{top3_names[1]}({w2r})×{top3_names[2]}({w3r})"
            r["type"]="3way"
            r["s1"]=top3_names[0];r["s2"]=top3_names[1];r["s3"]=top3_names[2]
            r["w1"]=w1r;r["w2"]=w2r;r["w3"]=w3r
            all_combos_3way.append(r)

    # ── 合并 ──
    all_combos=all_combos_2way+all_combos_3way

    # 最佳Sharpe
    best_sharpe=max(all_combos,key=lambda x:x["sharpe"])
    # 最佳Calmar
    best_calmar=max(all_combos,key=lambda x:x["calmar"])
    # 达标组合：回撤<20% 且 年化>10%
    qualified=[r for r in all_combos if abs(r["max_drawdown"])<0.20 and r["annual_return"]>0.10]
    qualified_sorted=sorted(qualified,key=lambda x:x["sharpe"],reverse=True)

    logger.info("\n"+"="*80)
    logger.info("结果摘要")
    logger.info("="*80)

    logger.info(f"\n子策略单独表现:")
    for sn in s_names:
        r=solo_results[sn]
        logger.info(f"  {sn:<16} Sharpe={r['sharpe']:.3f} 年化={r['annual_return']*100:>6.2f}% 回撤={abs(r['max_drawdown'])*100:.1f}% Calmar={r['calmar']:.3f}")

    logger.info(f"\n最佳Sharpe组合: {best_sharpe['name']}")
    logger.info(f"  Sharpe={best_sharpe['sharpe']:.3f} 年化={best_sharpe['annual_return']*100:.2f}% 回撤={abs(best_sharpe['max_drawdown'])*100:.1f}% Calmar={best_sharpe['calmar']:.3f}")

    logger.info(f"\n最佳Calmar组合: {best_calmar['name']}")
    logger.info(f"  Calmar={best_calmar['calmar']:.3f} 年化={best_calmar['annual_return']*100:.2f}% 回撤={abs(best_calmar['max_drawdown'])*100:.1f}% Sharpe={best_calmar['sharpe']:.3f}")

    logger.info(f"\n达标组合(回撤<20%,年化>10%):共{len(qualified)}个")
    header=f"{'排名':<4} {'组合名称':<55} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}"
    logger.info(header)
    logger.info("-"*len(header))
    for idx,r in enumerate(qualified_sorted[:20],1):
        logger.info(f"  {idx:<2} {r['name']:<55} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {abs(r['max_drawdown'])*100:>6.1f}% {r['calmar']:>7.3f}")

    # ── 保存 ──
    all_results={
        "data_info":{"nd":nd,"ns":ns},
        "solo":{sn:solo_results[sn] for sn in s_names},
        "top_sharpe":best_sharpe,
        "top_calmar":best_calmar,
        "qualified_count":len(qualified),
        "qualified_top20":qualified_sorted[:20],
        "all_combos_count":len(all_combos),
    }
    out_path=os.path.join(RESULTS_DIR,"results.json")
    with open(out_path,'w') as f:
        json.dump(all_results,f,indent=2,ensure_ascii=False,default=str)
    logger.info(f"\n结果已保存至: {out_path}")
    return all_results,out_path

if __name__=="__main__":
    main()
