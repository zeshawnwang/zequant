import pandas as pd
import numpy as np
from core.strategies.base.strategy import SignalStrategy
from core.screening.impl.factor_rank import FactorRankSelector
from core.screening.impl.multi_factor import MultiFactorSelector
from core.signals.base.composer import LayeredComposer, DirectComposer


def test_signal_strategy_name():
    strategy = SignalStrategy(name="test")
    assert strategy.name == "test"


def test_signal_strategy_default_top_n():
    strategy = SignalStrategy(name="test")
    assert strategy.top_n == 30


def test_signal_strategy_with_selector():
    selector = FactorRankSelector(factor_name="pe_ttm", ascending=True)
    strategy = SignalStrategy(name="test", selector=selector)
    assert strategy.selector is selector


def test_signal_strategy_custom_top_n():
    strategy = SignalStrategy(name="test", top_n=10)
    assert strategy.top_n == 10


def test_signal_strategy_with_composer():
    composer = LayeredComposer(top_n=5)
    strategy = SignalStrategy(name="test", composer=composer)
    assert strategy.composer is composer


def test_signal_strategy_no_selector():
    strategy = SignalStrategy(name="test", selector=None)
    assert strategy.selector is None


def test_signal_strategy_multi_factor_selector():
    weights = {"factor_a": 0.5, "factor_b": 0.5}
    selector = MultiFactorSelector(weights=weights, normalize_weights=True)
    strategy = SignalStrategy(name="test", selector=selector)
    assert strategy.selector.factor_names == ["factor_a", "factor_b"]


def test_signal_strategy_min_position():
    strategy = SignalStrategy(name="minpos", min_position=0.01)
    assert strategy.min_position == 0.01


def test_signal_strategy_risk_manager():
    strategy = SignalStrategy(name="risk", risk_manager="dummy")
    assert strategy.risk_manager == "dummy"
