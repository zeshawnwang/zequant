"""
V9 — 完整策略链路评估 + 7市场窗口分析

对所有已落地策略和达标实验做7个市场窗口的逐窗口评估。
"""
import os,sys,json,logging,gc,numpy as np,pandas as pd
from datetime import datetime

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger("v9")
TX=0.0012
FACTORS=list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position']))
V9_DIR=os.path.join(os.path.dirname(__file__),"v9_window_eval");os.makedirs(V9_DIR,exist_ok=True)

# 7个市场窗口
WINDOWS=[
    ("2019_修复牛","2019-01-02","2019-12-31"),
    ("2020_疫情冲击","2020-01-02","2020-12-31"),
    ("2021_结构牛","2021-01-04","2021-12-31"),
    ("2022_熊市","2022-01-04","2022-12-30"),
    ("2023_震荡修复","2023-01-03","2023-12-29"),
    ("2024_反弹","2024-01-02","2024-12-31"),
    ("2025_至今","2025-01-02","2026-04-30"),
]

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

def v1w():
    p=os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:return it['configs'][0]['weights']
    return {}

def bt_window(sig,fwd,dm,ds,name,rf=3,tn=40,pos_ratio=None):
    """回测全窗口 + 单窗口拆分。"""
    nd,ns=sig.shape
    alloc=RPPortfolioWeights(top_n=tn,min_hold_days=5)
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

    total=calc_metrics(eq,dr,nd,name,"全区间",nt)
    result=total

    # 每个窗口单独计算
    win_results=[total]
    for wname,ws,we in WINDOWS:
        try:
            ws_d=pd.Timestamp(ws);we_d=pd.Timestamp(we)
            in_window=[j for j,d in enumerate(ds) if ws_d<=d<=we_d]
            if len(in_window)<5:
                win_results.append({"window":wname,"annual_return":0,"sharpe":0,"max_drawdown":0,"calmar":0,"win_rate":0,"n_days":0})
                continue
            w_start=in_window[0];w_end=in_window[-1]+1
            w_eq=eq[w_start:w_end];w_dr=dr[w_start:w_end]
            w_dr[0]=0.0
            w_nt=nt  # 近似
            wr=calc_metrics(w_eq,w_dr,len(w_eq),name,wname,w_nt)
            wr["n_days"]=len(in_window)
            win_results.append(wr)
        except Exception as e:
            win_results.append({"window":wname,"error":str(e)})

    return result,win_results

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

def main():
    z3,fwd,dm,tks,fnames,nd,ns,ds=load()
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

    strategies=[
        ("mf_vol_d10_rp",mf,10,vol_p),
        ("mf_d10_rp",mf,10,None),
        ("mf_trend_d5_rp",mf,5,trend_p),
        ("v4_mf_tv_rp",mf,3,vol_p),
        ("v4_mf_rp",mf,3,None),
    ]

    all_window_data=[]
    for sname,sig,rf,pr in strategies:
        logger.info(f"\n{'='*60}\n评估: {sname}\n{'='*60}")
        result,win_results=bt_window(sig,fwd,dm,ds,sname,rf=rf,tn=40,pos_ratio=pr)
        for w in win_results:
            w["strategy"]=sname
            all_window_data.append(w)

    # 保存全窗口数据
    with open(os.path.join(V9_DIR,"window_results.json"),'w') as f:
        json.dump(all_window_data,f,indent=2,ensure_ascii=False)

    # 打印矩阵
    print(f"\n{'='*120}")
    print(f"{'策略':<24} {'窗口':<14} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6}")
    print('-'*120)
    for sname,_,_,_ in strategies:
        srows=[w for w in all_window_data if w['strategy']==sname]
        for w in srows:
            dd_str=f"{abs(w['max_drawdown'])*100:.1f}%" if w['max_drawdown']<0 else f"{w['max_drawdown']*100:.1f}%"
            print(f"  {sname:<22} {w['window']:<14} {w['annual_return']*100:>6.2f}% {w['sharpe']:>7.3f} {dd_str:>7} {w['calmar']:>7.3f} {w['win_rate']*100:>5.1f}%")
        print('-'*120)

    # 汇总: 哪些策略通过了窗口验证
    logger.info("\n=== 窗口通过汇总(正收益窗口/总窗口) ===")
    for sname,_,_,_ in strategies:
        srows=[w for w in all_window_data if w['strategy']==sname and w['window']!="全区间"]
        pos_wins=sum(1 for w in srows if w['annual_return']>0)
        total_win=len(srows)
        all_win_ok="✅" if pos_wins>=5 else("⚠️" if pos_wins>=3 else"❌")
        logger.info(f"  {all_win_ok} {sname:<22} {pos_wins}/{total_win} 窗口正收益")
        for w in srows:
            icon="+" if w['annual_return']>0 else" "
            logger.info(f"      {icon}{int(w['annual_return']*100):+3d}%  {w['window']}")

if __name__=="__main__":
    main()
