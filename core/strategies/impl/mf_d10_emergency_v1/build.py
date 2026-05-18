"""MF_D10_EMERGENCY_V1 — mf_d10_rp + 个股止损10% + 大盘熔断4%/3天恢复。

来源于 2026-05-18 紧急事件参数扫描（46个实验）：
- 最优止损阈值: 10%（sl=0.10）
- 最优熔断阈值: 4%（ct=0.04）
- 最优恢复天数: 3天（rd=3）

回测对比（修正Pipeline，2019-01~2026-04）：
                mf_d10_rp (原版)    mf_d10_emergency_v1    变化
  综合分          60.6               65.0                  +4.4
  年化            32.84%             35.00%                +2.16%
  Sharpe          1.208              1.303                 +0.095
  最大回撤         -37.60%            -31.24%               -6.36%
  2022熊市年化     +8.60%             +11.67%              +3.07%
  修复牛OOS Sharpe  2.459             2.562                 +0.103
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf_d10_emergency_v1(top_n: int = 20, **kwargs) -> SignalStrategy:
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
