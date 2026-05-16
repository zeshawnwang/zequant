"""
验证 StrategyPipeline 统一管道: 
- load / run / window_analysis / combine / export / from_config
"""
import os,sys,logging
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")

import numpy as np
from core.strategies.pipeline import StrategyPipeline,BacktestMetrics

# ── 信号构造: MF因子合成 (使用X6的v1w) ──
from X6_pipeline import v1w as x6_v1w

def mf_signal_builder(z3,fi,nd,ns):
    wv_dict=x6_v1w()
    wv=np.zeros(z3.shape[2],dtype=np.float32)
    fc_list=list(fi.keys())
    for fc_name,fc_weight in wv_dict.items():
        if fc_name in fi:
            wv[fi[fc_name]]=float(fc_weight)
    s=np.sum(np.abs(wv));wv/=s if s>0 else 1
    return np.nan_to_num(np.tensordot(z3,wv,axes=(2,0)),nan=-1e10,neginf=-1e10)

# ── 测试1: 基本管道 ──
print("="*60)
print("Test 1: 基础管道 (MF_D10) — 创建+load+run")
p=StrategyPipeline(
    signal_builder=mf_signal_builder,
    name="MF_D10",
    rebal_freq=10,top_n=50,min_hold_days=10,
    positioner_type='rp',tx_cost=0.0012,
)
r=p.run()
assert isinstance(r,BacktestMetrics), f"期望BacktestMetrics, 得到{type(r)}"
assert abs(r.sharpe-1.2)<0.5, f"Sharpe异常: {r.sharpe}"
print(f"  ✓ Sharpe={r.sharpe:.3f} 年化={r.annual_return*100:.2f}% 回撤={r.max_drawdown*100:.2f}%")

# ── 测试2: 窗口分析 ──
print("\n"+"="*60)
print("Test 2: 窗口分析")
windows=p.window_analysis()
assert len(windows)==8,f"期望8个窗口(1全区间+7子窗口),得到{len(windows)}"
print(f"  ✓ {len(windows)}个窗口分析完成")
for w in windows[:3]:
    print(f"    {w.window:<12} 年化={w.annual_return*100:>+6.2f}% Sharpe={w.sharpe:.3f}")

# ── 测试3: 限区间运行 ──
print("\n"+"="*60)
print("Test 3: 限区间运行 (2020-2021)")
r2=p.run(start="2020-01-02",end="2021-12-31")
print(f"  ✓ Sharpe={r2.sharpe:.3f} 年化={r2.annual_return*100:.2f}% 回撤={r2.max_drawdown*100:.2f}%")
assert abs(r2.annual_return)>0.01

# ── 测试4: combine ──
print("\n"+"="*60)
print("Test 4: combine 策略组合")
p2=StrategyPipeline(
    signal_builder=lambda z3,fi,nd,ns:np.ones((nd,ns),dtype=np.float32)*-0.5,
    name="Dummy_Short",
    rebal_freq=3,top_n=40,
)
p2.load();p2.build_signal()
combo_metrics=p.combine(p2,weight=0.7)
print(f"  ✓ 组合: {combo_metrics.name} Sharpe={combo_metrics.sharpe:.3f}")

# ── 测试5: export ──
print("\n"+"="*60)
print("Test 5: export")
out=p.export("pipeline_test")
assert os.path.exists(out),f"导出文件不存在: {out}"
print(f"  ✓ 已导出至: {out}")

# ── 测试6: from_config ──
print("\n"+"="*60)
print("Test 6: from_config")
cfg={"name":"cfg_test","rebal_freq":5,"top_n":30,"positioner_type":"covrp","tx_cost":0.001}
p3=StrategyPipeline.from_config(cfg)
assert p3.name=="cfg_test"
assert p3.rebal_freq==5
assert p3.positioner_type=="covrp"
print(f"  ✓ from_config → {p3}")

# ── 测试7: combo_from_series (静态方法) ──
print("\n"+"="*60)
print("Test 7: combo_from_series 静态方法")
dr1=np.random.randn(500)*0.01;dr2=np.random.randn(500)*0.01
cr=StrategyPipeline.combo_from_series(dr1,dr2,w1=0.6,w2=0.4)
assert "sharpe" in cr,"combo_from_series返回缺少sharpe"
assert "annual_return" in cr
print(f"  ✓ combo_from_series: Sharpe={cr['sharpe']:.3f} 年化={cr['annual_return']*100:.2f}%")

# ── 总结 ──
print("\n"+"="*60)
print("所有测试通过! StrategyPipeline 完整可用。")
print("="*60)
