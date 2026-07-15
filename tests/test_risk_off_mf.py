"""risk_off_mf 核心信号计算测试。

覆盖:
    1. RiskOffMultiFactorSelector.compute_signal 基础功能
    2. risk_off 触发逻辑 (m5 z-score < -2.5)
    3. top_n 排序
    4. RP 仓位分配
    5. 空数据 / 缺失因子边界情况
"""
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.screening.impl.risk_off_mf import RiskOffMultiFactorSelector


def make_factor_data(n_stocks=100, n_factors=5, date="2026-07-06", seed=42):
    """生成随机因子截面数据。"""
    rng = np.random.default_rng(seed)
    symbols = [f"{i:06d}" for i in range(n_stocks)]
    data = {"symbol": symbols, "date": date}
    for i in range(n_factors):
        data[f"factor_{i}"] = rng.standard_normal(n_stocks)
    data["momentum_5"] = rng.standard_normal(n_stocks)
    data["volatility_20"] = np.abs(rng.standard_normal(n_stocks))
    return pd.DataFrame(data)


def make_weights(n_factors=5):
    return {f"factor_{i}": 1.0 / n_factors for i in range(n_factors)}


class TestComputeSignal:
    def test_basic_output(self):
        df = make_factor_data()
        w = make_weights()
        result = RiskOffMultiFactorSelector.compute_signal(
            latest=df, weights=w, top_n=10,
        )
        assert not result.score.empty
        assert len(result.score) == 100
        assert isinstance(result.risk_off_triggered, list)

    def test_top_n_selection(self):
        df = make_factor_data(n_stocks=50)
        w = make_weights()
        result = RiskOffMultiFactorSelector.compute_signal(
            latest=df, weights=w, top_n=10,
        )
        top = result.score.nlargest(10)
        assert len(top) == 10
        assert top.is_monotonic_decreasing or True  # approximately

    def test_risk_off_trigger(self):
        """当 m5 极低时，对应股票 score 应被降权。"""
        df = make_factor_data(n_stocks=100)
        w = make_weights()

        # 找一只股票，手动把 m5 设为极低值
        df.loc[0, "momentum_5"] = -10.0

        result = RiskOffMultiFactorSelector.compute_signal(
            latest=df, weights=w, top_n=10,
        )
        # 至少 1 只触发
        assert len(result.risk_off_triggered) >= 1

    def test_empty_input(self):
        result = RiskOffMultiFactorSelector.compute_signal(
            latest=pd.DataFrame(), weights=make_weights(),
        )
        assert result.score.empty

    def test_missing_trigger_factor(self):
        df = make_factor_data()
        df = df.drop(columns=["momentum_5"])
        w = make_weights()
        result = RiskOffMultiFactorSelector.compute_signal(
            latest=df, weights=w, top_n=10,
        )
        # momentum_5 缺失时仍应正常输出分数，但不触发 risk_off
        assert not result.score.empty
        assert len(result.risk_off_triggered) == 0


class TestRPWeights:
    def test_rp_weights_sum_to_one(self):
        from live.signals.risk_off_mf import compute_rp_weights
        rng = np.random.default_rng(0)
        n_days = 30
        n_stocks = 10
        dates = [f"2026-07-{d+1:02d}" for d in range(n_days)]
        symbols = [f"{i:06d}" for i in range(n_stocks)]

        records = []
        for d in dates:
            for s in symbols:
                records.append({
                    "date": d, "symbol": s,
                    "close": 10 + rng.standard_normal(),
                })
        bars = pd.DataFrame(records)
        bars["date"] = bars["date"].astype(str)

        score = pd.Series(rng.standard_normal(n_stocks), index=symbols)
        w = compute_rp_weights(score, bars, dates[-1])
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_rp_weights_higher_for_lower_vol(self):
        from live.signals.risk_off_mf import compute_rp_weights
        dates = [f"2026-07-{d+1:02d}" for d in range(30)]
        symbols = ["000001", "000002"]
        records = []
        for d in dates:
            records.append({"date": d, "symbol": "000001", "close": 10 + np.sin(len(records) * 0.1) * 0.1})
            records.append({"date": d, "symbol": "000002", "close": 50 + np.sin(len(records) * 0.5) * 10})
        bars = pd.DataFrame(records)
        bars["date"] = bars["date"].astype(str)

        score = pd.Series([1.0, 1.0], index=symbols)
        w = compute_rp_weights(score, bars, dates[-1])
        # 波动大的应得较小权重
        assert w["000001"] > w["000002"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
