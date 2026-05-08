"""StrategyHub 注册/创建冒烟测试。

验证:
  - import strategies 触发后,内置 5 个策略可被发现
  - 静态策略 momentum_top50 / low_vol_top50 可创建
  - alpha101_walk_forward 在缺 eval_summary 时抛错(契约保护)
  - meta.eval_factor_filter 在 alpha101 路径上声明为 'alpha'
"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

import strategies  # noqa: F401  触发注册
from core.strategy_hub import create, get_meta, list_all, list_by_category, categories, describe
from core.strategy import QuantStrategy


REQUIRED_STRATEGIES = (
    "momentum_top50",
    "low_vol_top50",
    "alpha101_manual",
    "alpha101_from_registry",
    "alpha101_walk_forward",
)


def test_all_required_strategies_registered():
    names = list_all()
    for s in REQUIRED_STRATEGIES:
        assert s in names, f"strategy not registered: {s}"


def test_create_static_strategy_momentum():
    strat = create("momentum_top50", top_n=20)
    assert isinstance(strat, QuantStrategy)
    assert strat.top_n == 20
    assert hasattr(strat.selector, "select")


def test_create_static_strategy_low_vol():
    strat = create("low_vol_top50", top_n=15)
    assert isinstance(strat, QuantStrategy)
    assert strat.top_n == 15


def test_alpha101_walk_forward_requires_eval_summary():
    with pytest.raises(ValueError):
        create("alpha101_walk_forward", db=None,
               eval_summary=None, top_n=10)


def test_alpha101_walk_forward_meta_filter():
    meta = get_meta("alpha101_walk_forward")
    assert meta.requires_evaluation is True
    assert meta.eval_factor_filter == "alpha"
    assert meta.timing_factors  # 非空