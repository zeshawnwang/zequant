import pandas as pd
import numpy as np
from core.execution.impl.backtest import BacktestEngine, BacktestReport
from core.strategies.base.strategy import SignalStrategy
from core.screening.impl.factor_rank import FactorRankSelector
from core.signals.base.composer import LayeredComposer


def _make_factor_data(n_symbols=10, n_days=30, seed=42):
    np.random.seed(seed)
    symbols = [f"STOCK{i:04d}" for i in range(n_symbols)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    records = []
    for sym in symbols:
        price = 30.0
        for d in dates:
            open_p = price * (1 + np.random.normal(0, 0.005))
            close = open_p * (1 + np.random.normal(0, 0.02))
            factor_a = np.random.randn()
            records.append({
                "symbol": sym, "date": d,
                "open": round(open_p, 2), "close": round(close, 2),
                "factor_a": factor_a, "factor_value": -factor_a if "0" in sym else factor_a,
            })
            price = close
    return pd.DataFrame(records)


def test_backtest_engine_creation():
    engine = BacktestEngine()
    assert engine.initial_capital == 1_000_000


def test_backtest_engine_run_empty():
    engine = BacktestEngine()
    strategy = SignalStrategy(name="test", selector=None)
    data = pd.DataFrame(columns=["date", "symbol", "close"])
    report = engine.run(strategy, data, "2024-01-01", "2024-01-10")
    assert isinstance(report, BacktestReport)


def test_backtest_with_selector():
    data = _make_factor_data(n_symbols=10, n_days=20)
    selector = FactorRankSelector(factor_name="factor_a", ascending=False)
    strategy = SignalStrategy(
        name="test_selector",
        selector=selector,
        composer=LayeredComposer(top_n=5),
        top_n=5,
    )
    engine = BacktestEngine()
    report = engine.run(strategy, data, "2024-01-01", "2024-01-20")
    assert isinstance(report, BacktestReport)
    assert report.sharpe_ratio is not None


def test_backtest_missing_date_column():
    engine = BacktestEngine()
    strategy = SignalStrategy(name="test")
    bad_data = pd.DataFrame({"symbol": ["A"], "close": [10.0]})
    try:
        engine.run(strategy, bad_data, "2024-01-01", "2024-01-10")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "date" in str(e)


def test_backtest_report_metrics():
    data = _make_factor_data(n_symbols=5, n_days=15, seed=99)
    selector = FactorRankSelector(factor_name="factor_value", ascending=True)
    strategy = SignalStrategy(
        name="metric_test",
        selector=selector,
        composer=LayeredComposer(top_n=3),
        top_n=3,
    )
    engine = BacktestEngine(initial_capital=500_000)
    report = engine.run(strategy, data, "2024-01-01", "2024-01-15")
    assert report.total_return is not None
    assert report.annualized_return is not None
    assert report.sharpe_ratio is not None
    assert report.max_drawdown is not None
    assert report.win_rate is not None


def test_backtest_strategy_last_selected():
    data = _make_factor_data(n_symbols=6, n_days=10, seed=42)
    selector = FactorRankSelector(factor_name="factor_a", ascending=False)
    strategy = SignalStrategy(
        name="sel_test",
        selector=selector,
        composer=LayeredComposer(top_n=3),
        top_n=3,
    )
    engine = BacktestEngine()
    engine.run(strategy, data, "2024-01-01", "2024-01-10")
    assert hasattr(strategy, "last_selected")
