"""MF+双周+RP 2026-05-17修正版 — 真实费率+Universe过滤验证。

来源于 2026-05-17 MF全参数扫描（72个实验），确认：
- top_n=20 最佳（优于 30/40/50）
- rebal_freq=10 最佳（优于 3/5）
- Universe过滤+真实tx_cost=0.002 后仍三区间通过

修正Pipeline下的验证结果：
  全区间: 年化32.84% Sharpe=1.208 回撤-37.60%
  2022熊市: 年化+8.60%（三区间通过）
  修复牛OOS: Sharpe=2.459（泛化验证通过）
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf_d10_opt_0517(top_n: int = 20, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = MultiFactorSelector(weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01), top_n=top_n, normalize_weights=True)
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=None, composer=composer, risk_manager=risk, top_n=top_n)
