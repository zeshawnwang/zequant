"""策略构建工厂 — 共享 boilerplate 逻辑。"""
from __future__ import annotations
import json
import os
from ..base.strategy import SignalStrategy
from ...signals import LayeredComposer, MaxSingleWeightConstraint
from ...risk import RiskManager


def _build_signal_strategy(
    cfg_dir: str,
    selector,
    top_n: int = 20,
    position_sizer=None,
    name: str | None = None,
) -> SignalStrategy:
    """构建 SignalStrategy 的共享工厂函数。

    参数
    ----------
    cfg_dir : str
        策略目录路径（用于加载 config.json）。
    selector :
        已配置的 selector 实例。
    top_n : int
        选股数量上限。
    position_sizer :
        仓位调整器（可选，默认 None）。
    name : str | None
        策略名称（默认从 config.json 读取 strategy.name）。
    """
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)

    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(
                max_weight=cfg["composer"]["constraints"][0]["max_single_weight"]
            )
        ],
    )
    risk = RiskManager(config=cfg.get("risk", {}))
    strategy_name = name or cfg["strategy"]["name"]

    return SignalStrategy(
        name=strategy_name,
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk,
        top_n=top_n,
    )
