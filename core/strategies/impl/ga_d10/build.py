"""GA优化+10d+RP — 全量GA+新技术因子优化结果。

X5实验GA(rf=10)结果：年化33.59%, Sharpe 1.435, 回撤-16.33%。

核心：MultiFactorSelector(GA优化66因子权重) + 双周调仓(10d) + 风险平价。
全量GA最高Sharpe(1.435) + 回撤最低(-16.33%)，唯一同时满足回撤<20%+Sharpe>1.4的策略。
"""
from __future__ import annotations
import json, os
from ....screening import MultiFactorSelector
from .._factory import _build_signal_strategy

def build_ga_d10(top_n: int = 40, **kwargs):
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = MultiFactorSelector(weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01), top_n=top_n, normalize_weights=True)
    return _build_signal_strategy(cfg_dir, selector, top_n=top_n)
