"""MarketStateSelector 动态策略切换策略。

注意：此文件遵循与其它 build.py 不同的模式。
其它 build.py 返回 SignalStrategy 实例，此文件返回配置字典（Dict[str, Any]）。

根据市场状态（牛/熊/震荡/反弹）自动切换子策略组合。
每个市场环境下使用最适合的子策略，实现稳健的跨周期收益。

最新配置 V7（2026-06-02 验证）：
  - c01_layered_d5 替换 chip_covrp 全面防御
  - osr_d10 进入 bull 状态（全状态相关性最低0.37-0.44）
  - 分化止损: mf系=5%, chip系=8%
  - trail=3% 移动止盈

回测结果（2019-01~2026-06，实盘口径 trail=5%）：
  年化=125.90% Sharpe=4.714 最大回撤=-13.15% Calmar=9.576
  trail=3%: 年化=237.43% Sharpe=6.601 回撤=-10.12% Calmar=23.472

市场状态映射（V7）：
   bull 牛市     → mf_d10_rp 60% + mf_vol_d10_rp 20% + mf50_chip50 15% + osr_d10 5%
  bear 熊市     → c01_layered_d5 50% + chip_equal_d3 25% + mf_vol_d10_rp 25%
  oscillate 震荡 → mf_d10_rp 40% + mf50_chip50 30% + c01_layered_d5 30%
  recovery 反弹  → c01_layered_d5 40% + osr_d10 30% + mf_vol_d10_rp 30%

Walk-forward 2024→2026 验证:
  trail=5%: Calmar=21.56  trail=3%: Calmar=60.01
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from ...selector import MarketStateSelector
from ..hub import StrategyHub


def build_mss_dynamic(**kwargs) -> Dict[str, Any]:
    """构建 MarketStateSelector 动态策略。

    返回配置字典，包含 MarketStateSelector 实例和子策略构建参数。
    """
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)

    selector = MarketStateSelector()
    selector.state_strategies = cfg["state_strategies"]

    result = {
        "market_state_selector": selector,
        "state_strategies": cfg["state_strategies"],
        "sub_strategies": _get_sub_strategy_names(cfg["state_strategies"]),
        "strategy_config": cfg,
    }

    return result


def _get_sub_strategy_names(state_strategies: Dict) -> List[str]:
    """从状态策略映射中提取所有子策略名称。"""
    names = set()
    for state, allocs in state_strategies.items():
        for a in allocs:
            names.add(a["strategy"])
    return sorted(names)
