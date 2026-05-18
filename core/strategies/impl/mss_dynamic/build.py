"""MarketStateSelector 动态策略切换策略。

根据市场状态（牛/熊/震荡/反弹）自动切换子策略组合。
每个市场环境下使用最适合的子策略，实现稳健的跨周期收益。

最新配置 V6a_3way（推荐）：
  - 每状态3个子策略，更均衡的多样化
  - 用 chip_equal_d3 替代 osr_d10（osr_d10回撤-50%）
  - 用 chip_equal_d3/c01_layered_d5 增加快速恢复能力

回测结果（2019-01~2026-04，9/9窗口正收益）：
  年化=24.75% Sharpe=1.421 最大回撤=-13.84%
  回撤修复仅9天（V5原版86天 → 提升9.5倍）

市场状态映射（V6a_3way）：
  bull 牛市     → mf_d10_rp 60% + mf_vol_d10_rp 20% + chip_covrp 20%
  bear 熊市     → chip_covrp 60% + chip_equal_d3 20% + mf_vol_d10_rp 20%
  oscillate 震荡 → chip_covrp 40% + mf50_chip50 30% + c01_layered_d5 30%
  recovery 反弹  → chip_equal_d3 40% + mf60_chip40 30% + mf_vol_d10_rp 30%

另存可选配置 V5_original（更高Sharpe）:
  bull 牛市 → mf_d10_rp 70% + chip_rp 30%
  bear 熊市 → chip_covrp 70% + mf_vol_d10_rp 30%
  oscillate 震荡 → mf50_chip50 50% + chip_covrp 50%
  recovery 反弹 → mf60_chip40 60% + osr_d10 40%
  年化=26.92% Sharpe=1.500 回撤=-13.16% 修复=86天
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
