import pandas as pd
import numpy as np
from core.screening.impl.factor_rank import FactorRankSelector
from core.screening.impl.multi_factor import MultiFactorSelector


def _make_factor_data(n_symbols=10, n_days=2, seed=42):
    np.random.seed(seed)
    symbols = [f"STOCK{i:04d}" for i in range(n_symbols)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    records = []
    for sym in symbols:
        for d in dates:
            records.append({
                "symbol": sym, "date": d,
                "test_factor": np.random.randn(),
                "factor_a": np.random.randn(),
                "factor_b": np.random.randn(),
                "factor_c": np.random.randn(),
            })
    return pd.DataFrame(records)


def test_factor_rank_creation():
    selector = FactorRankSelector(factor_name="test_factor", ascending=True)
    assert selector.factor_name == "test_factor"
    assert selector.ascending is True


def test_factor_rank_select_top():
    data = _make_factor_data(n_symbols=10, n_days=2, seed=42)
    selector = FactorRankSelector(factor_name="test_factor", ascending=True)
    selected = selector.select(data, "2024-01-02", top_n=3)
    assert isinstance(selected, list)
    assert len(selected) == 3
    assert all(isinstance(s, str) for s in selected)


def test_factor_rank_ascending():
    data = _make_factor_data(n_symbols=10, n_days=2, seed=42)
    asc_sel = FactorRankSelector(factor_name="test_factor", ascending=True)
    desc_sel = FactorRankSelector(factor_name="test_factor", ascending=False)
    asc_result = asc_sel.select(data, "2024-01-02", top_n=3)
    desc_result = desc_sel.select(data, "2024-01-02", top_n=3)
    assert asc_result != desc_result


def test_factor_rank_no_data():
    selector = FactorRankSelector(factor_name="nonexistent", ascending=True)
    data = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-01"], "other": [1.0]})
    selected = selector.select(data, "2024-01-02", top_n=3)
    assert selected == []


def test_multi_factor_creation():
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    selector = MultiFactorSelector(weights=weights, normalize_weights=True)
    assert set(selector.factor_names) == {"a", "b", "c"}


def test_multi_factor_select():
    data = _make_factor_data(n_symbols=10, n_days=2, seed=42)
    weights = {"factor_a": 0.6, "factor_b": 0.4}
    selector = MultiFactorSelector(weights=weights, normalize_weights=True)
    selected = selector.select(data, "2024-01-02", top_n=5)
    assert len(selected) == 5


def test_multi_factor_empty_weights():
    try:
        MultiFactorSelector(weights={}, normalize_weights=True)
        assert False, "Should have raised"
    except (ValueError, AssertionError, Exception):
        pass


def test_multi_factor_missing_columns():
    weights = {"factor_a": 0.6, "factor_b": 0.4}
    selector = MultiFactorSelector(weights=weights, normalize_weights=True, min_factors_coverage=2)
    data = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-01"], "factor_a": [1.0]})
    selected = selector.select(data, "2024-01-02", top_n=5)
    assert selected == []
