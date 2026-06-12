"""MF+双周+RP — 高收益低频策略。

V6实验X01_MF+无择时+D10结果：年化38.03%, Sharpe 1.306, 回撤-30.44%。

核心：MultiFactorSelector(V1权重) + 双周调仓(10d) + 风险平价。最高Sharpe的基础策略。
"""
from __future__ import annotations
import json, os
from ....screening import MultiFactorSelector
from .._factory import _build_signal_strategy

def build_mf_d10_rp(top_n: int = 20, **kwargs):
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = MultiFactorSelector(weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01), top_n=top_n, normalize_weights=True)
    return _build_signal_strategy(cfg_dir, selector, top_n=top_n)
