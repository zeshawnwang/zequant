#!/usr/bin/env python3
"""因子-策略适配性分析脚本。

评估不同因子在不同策略下的表现，找出最优匹配。
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np

from core.config import load_config, get_db_path
from core.database import Database
from core.backtest import BacktestEngine
from core.universe import SymbolUniverse, UniverseConfig
from core.factor_evaluator import FactorEvaluator
from screening.factor_rank import FactorRankSelector
from timings.trend import TrendTiming
from portfolios.equal_weight import EqualWeightBuilder
from portfolios.risk_parity import RiskParityBuilder
from core.strategy import QuantStrategy
import factors  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zequant.analysis")


@dataclass
class BacktestResult:
    """回测结果。"""
    factor: str
    strategy: str
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int


class FactorStrategyAnalyzer:
    """因子-策略适配性分析器。"""

    def __init__(self, db: Database, cfg: Dict):
        self.db = db
        self.cfg = cfg
        self.results: List[BacktestResult] = []

    def evaluate_factors(self, start: str, end: str, forward_days: int = 5) -> pd.DataFrame:
        """评估所有因子。"""
        logger.info("评估因子: %s ~ %s", start, end)

        all_names = self.db.list_factor_columns()
        alpha_names = sorted(
            [n for n in all_names if n.startswith("a") and n[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
        tech_names = [n for n in all_names if n in {
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_5", "volatility_10", "volatility_20",
            "rsi_14", "macd", "volume_ratio", "boll_position",
        }]

        factor_names = alpha_names + tech_names
        logger.info("评估 %d 个因子...", len(factor_names))

        ev = FactorEvaluator(self.db)
        summary = ev.evaluate_all(
            factor_names=factor_names,
            start_date=start,
            end_date=end,
            forward_days=forward_days,
        )

        return summary

    def _create_engine(self):
        """创建回测引擎。"""
        universe_cfg = UniverseConfig.from_config(self.cfg.get("universe", {}))
        universe = SymbolUniverse(self.db, universe_cfg)
        return BacktestEngine(
            initial_capital=self.cfg["backtest"]["initial_capital"],
            fee_config=self.cfg.get("fees", {}),
            risk_config=self.cfg.get("risk", {}),
            universe=universe,
        )

    def _load_factor_data(self, bt_start: str, bt_end: str, factor_names: List[str]) -> pd.DataFrame:
        """加载因子数据。"""
        available = set(self.db.list_factor_columns())
        wanted = set(factor_names) | {"momentum_5", "momentum_20", "macd", "rsi_14"}
        factor_cols = sorted([c for c in wanted if c in available])
        factor_data = self.db.get_factors(
            start_date=bt_start, end_date=bt_end,
            factor_names=factor_cols if factor_cols else None,
            with_close=True,
        )
        if factor_data.empty:
            return factor_data
        bars = self.db.get_daily_bars(start_date=bt_start, end_date=bt_end)
        if not bars.empty and "open" in bars.columns:
            open_data = bars[["date", "symbol", "open"]].copy()
            open_data["date"] = pd.to_datetime(open_data["date"])
            factor_data["date"] = pd.to_datetime(factor_data["date"])
            factor_data = factor_data.merge(open_data, on=["date", "symbol"], how="left")
        return factor_data

    def _run_backtest(
        self,
        factor: str,
        strategy_name: str,
        ascending: bool,
        factor_data: pd.DataFrame,
        bt_start: str,
        bt_end: str,
        top_n: int = 30,
        timing_params: Dict = None,
    ) -> BacktestResult:
        """运行单次回测。"""
        try:
            engine = self._create_engine()

            selector = FactorRankSelector(
                factor_name=factor,
                ascending=ascending,
                top_n=top_n * 3,
            )

            tp = timing_params or {}
            timing = TrendTiming(
                sma_short=tp.get("sma_short", 5),
                sma_medium=tp.get("sma_medium", 20),
                buy_threshold=tp.get("buy_threshold", 0.6),
                sell_threshold=tp.get("sell_threshold", 0.4),
            )

            portfolio = EqualWeightBuilder(reserve_cash_ratio=0.1)

            strategy = QuantStrategy(
                name=f"{factor}_{strategy_name}",
                selector=selector,
                timing=timing,
                portfolio=portfolio,
                top_n=top_n,
            )

            report = engine.run(
                strategy=strategy,
                factor_data=factor_data,
                start_date=bt_start,
                end_date=bt_end,
                rebalance_freq="1d",
            )

            return BacktestResult(
                factor=factor,
                strategy=strategy_name,
                total_return=report.total_return,
                annualized_return=report.annualized_return,
                max_drawdown=report.max_drawdown,
                sharpe_ratio=report.sharpe_ratio,
                total_trades=report.total_trades,
            )
        except Exception as e:
            logger.warning("回测失败 [%s, %s]: %s", factor, strategy_name, e)
            return None

    def test_factor_strategy_combinations(
        self,
        factors: List[str],
        ir_dict: Dict[str, float],
        bt_start: str,
        bt_end: str,
    ) -> List[BacktestResult]:
        """测试因子-策略组合。"""
        logger.info("=" * 60)
        logger.info("测试因子-策略组合")
        logger.info("=" * 60)

        results = []
        factor_data = self._load_factor_data(bt_start, bt_end, factors)
        if factor_data.empty:
            logger.error("无因子数据")
            return results

        for i, factor in enumerate(factors):
            ir = ir_dict.get(factor, 0)
            ascending = ir < 0

            logger.info("[%d/%d] 测试因子: %s (IR=%.3f, %s)",
                       i+1, len(factors), factor, ir, "反转" if ascending else "趋势")

            # 测试不同择时参数
            timing_configs = [
                {"name": "默认", "sma_short": 5, "sma_medium": 20, "buy_threshold": 0.6, "sell_threshold": 0.4},
                {"name": "短期", "sma_short": 3, "sma_medium": 10, "buy_threshold": 0.6, "sell_threshold": 0.4},
                {"name": "保守", "sma_short": 5, "sma_medium": 20, "buy_threshold": 0.7, "sell_threshold": 0.3},
            ]

            for tc in timing_configs:
                result = self._run_backtest(
                    factor=factor,
                    strategy_name=tc["name"],
                    ascending=ascending,
                    factor_data=factor_data,
                    bt_start=bt_start,
                    bt_end=bt_end,
                    timing_params=tc,
                )
                if result:
                    results.append(result)

        return results

    def generate_report(self, results: List[BacktestResult], factor_summary: pd.DataFrame):
        """生成适配性报告。"""
        if not results:
            print("\n无有效结果")
            return

        df = pd.DataFrame([{
            "因子": r.factor,
            "策略": r.strategy,
            "总收益": f"{r.total_return*100:.2f}%",
            "年化": f"{r.annualized_return*100:.2f}%",
            "最大回撤": f"{r.max_drawdown*100:.2f}%",
            "夏普": f"{r.sharpe_ratio:.2f}",
            "交易数": r.total_trades,
        } for r in results])

        print("\n" + "=" * 100)
        print("因子-策略适配性分析报告")
        print("=" * 100)

        print("\n【1. 因子评估结果 Top 20】")
        top_factors = factor_summary.head(20)[["factor_name", "ic_mean", "ir", "turnover", "monotonic"]]
        top_factors.columns = ["因子", "IC均值", "IR", "换手率", "单调性"]
        print(top_factors.to_string(index=False))

        print("\n【2. 因子-策略适配矩阵（按夏普比率）】")
        pivot = df.pivot_table(index="因子", columns="策略", values="夏普", aggfunc="first")
        print(pivot.to_string())

        print("\n【3. 每个因子的最优策略】")
        best_by_factor = df.loc[df.groupby("因子")["夏普"].idxmax()]
        best_by_factor = best_by_factor.sort_values("夏普", ascending=False)
        print(best_by_factor.to_string(index=False))

        print("\n【4. Top 10 最优组合】")
        top10 = df.sort_values("夏普", ascending=False).head(10)
        print(top10.to_string(index=False))

        print("\n【5. 最优匹配总结】")
        best = df.sort_values("夏普", ascending=False).iloc[0]
        print(f"最优因子: {best['因子']}")
        print(f"最优策略: {best['策略']}")
        print(f"总收益: {best['总收益']}")
        print(f"年化收益: {best['年化']}")
        print(f"最大回撤: {best['最大回撤']}")
        print(f"夏普比率: {best['夏普']}")

        print("\n" + "=" * 100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="因子-策略适配性分析")
    parser.add_argument("--eval-start", default="2020-01-01")
    parser.add_argument("--eval-end", default="2020-06-30")
    parser.add_argument("--bt-start", default="2020-07-01")
    parser.add_argument("--bt-end", default="2024-12-31")
    parser.add_argument("--top-factors", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config("config/config.yaml")
    db = Database(get_db_path(cfg))
    analyzer = FactorStrategyAnalyzer(db, cfg)

    # 因子评估
    summary = analyzer.evaluate_factors(
        start=args.eval_start,
        end=args.eval_end,
        forward_days=5,
    )

    if summary.empty:
        print("因子评估失败")
        db.close()
        return

    # 选择 Top N 因子
    top_factors = summary.head(args.top_factors)
    factor_list = top_factors["factor_name"].tolist()
    ir_dict = dict(zip(summary["factor_name"], summary["ir"]))

    print(f"\n选择 Top {len(factor_list)} 因子进行策略测试")
    print(f"因子列表: {', '.join(factor_list)}")

    # 测试因子-策略组合
    results = analyzer.test_factor_strategy_combinations(
        factors=factor_list,
        ir_dict=ir_dict,
        bt_start=args.bt_start,
        bt_end=args.bt_end,
    )

    # 生成报告
    analyzer.generate_report(results, summary)

    db.close()


if __name__ == "__main__":
    main()
