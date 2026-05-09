#!/usr/bin/env python3
"""参数优化框架：寻找策略、选股、因子、择时的最优匹配。

优化流程
--------
1. 因子评估：评估所有因子，筛选 IR 最高的因子
2. 策略对比：对不同策略进行快速基线测试
3. 择时参数网格搜索：对择时参数进行网格搜索
4. Walk-forward 验证：用滚动窗口验证最优配置的稳定性

用法
----
# 快速扫描（使用小数据集）
python3 scripts/optimize_params.py --mode quick

# 完整扫描
python3 scripts/optimize_params.py --mode full \
    --eval-start 2020-01-01 --eval-end 2020-06-30 \
    --bt-start 2020-07-01 --bt-end 2024-12-31

# 择时参数网格搜索
python3 scripts/optimize_params.py --mode timing_grid \
    --eval-start 2020-01-01 --eval-end 2020-06-30 \
    --bt-start 2020-07-01 --bt-end 2024-12-31
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.config import load_config, get_db_path
from core.database import Database
from core.backtest import BacktestEngine
from core.universe import SymbolUniverse, UniverseConfig
from core.factor_evaluator import FactorEvaluator
from screening.factor_rank import FactorRankSelector
from timings.trend import TrendTiming
from portfolios.equal_weight import EqualWeightBuilder
from core.strategy import QuantStrategy
import strategies  # noqa: F401
import factors      # noqa: F401

logger = logging.getLogger("zequant.optimize")


@dataclass
class BacktestResult:
    """单次回测结果。"""
    config: Dict
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    final_value: float


class Optimizer:
    """参数优化框架。"""

    def __init__(self, db: Database, cfg: Dict):
        self.db = db
        self.cfg = cfg
        self.results: List[BacktestResult] = []

    def _create_engine(self):
        """创建回测引擎。"""
        universe_cfg = UniverseConfig.from_config(self.cfg.get("universe", {}))
        universe = SymbolUniverse(self.db, universe_cfg)
        initial_capital = self.cfg["backtest"]["initial_capital"]
        return BacktestEngine(
            initial_capital=initial_capital,
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

    def _run_single_backtest(
        self,
        strategy_config: Dict,
        factor_data: pd.DataFrame,
        bt_start: str,
        bt_end: str,
    ) -> Optional[BacktestResult]:
        """运行单次回测。"""
        try:
            engine = self._create_engine()

            selector = FactorRankSelector(
                factor_name=strategy_config["factor_name"],
                ascending=strategy_config.get("ascending", False),
                top_n=strategy_config["top_n"] * 3,
            )

            timing = TrendTiming(
                sma_short=strategy_config.get("sma_short", 5),
                sma_medium=strategy_config.get("sma_medium", 20),
                buy_threshold=strategy_config.get("buy_threshold", 0.6),
                sell_threshold=strategy_config.get("sell_threshold", 0.4),
            )

            portfolio = EqualWeightBuilder(
                reserve_cash_ratio=strategy_config.get("reserve_cash_ratio", 0.1),
            )

            strategy = QuantStrategy(
                name="OptimizedStrategy",
                selector=selector,
                timing=timing,
                portfolio=portfolio,
                top_n=strategy_config["top_n"],
            )

            report = engine.run(
                strategy=strategy,
                factor_data=factor_data,
                start_date=bt_start,
                end_date=bt_end,
                rebalance_freq=self.cfg["backtest"].get("rebalance_freq", "1d"),
            )

            return BacktestResult(
                config=strategy_config,
                total_return=report.total_return,
                annualized_return=report.annualized_return,
                max_drawdown=report.max_drawdown,
                sharpe_ratio=report.sharpe_ratio,
                win_rate=report.win_rate,
                profit_factor=report.profit_factor,
                total_trades=report.total_trades,
                final_value=report.final_value,
            )
        except Exception as e:
            logger.warning("回测失败: %s", e)
            return None

    def evaluate_factors(
        self,
        eval_start: str,
        eval_end: str,
        forward_days: int = 5,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """第1层：因子评估。"""
        logger.info("=" * 60)
        logger.info("第1层：因子评估")
        logger.info("=" * 60)

        all_names = self.db.list_factor_columns()
        alpha_names = sorted(
            [n for n in all_names if n.startswith("a") and n[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
        tech_names = [n for n in all_names if n in {
            "momentum_5", "momentum_10", "momentum_20", "momentum_60",
            "volatility_5", "volatility_10", "volatility_20",
            "rsi_14", "macd", "volume_ratio", "boll_position",
        }]

        factor_names = alpha_names + tech_names
        logger.info("评估 %d 个因子...", len(factor_names))

        ev = FactorEvaluator(self.db)
        summary = ev.evaluate_all(
            factor_names=factor_names,
            start_date=eval_start,
            end_date=eval_end,
            forward_days=forward_days,
        )

        top_factors = summary.head(top_n)
        logger.info("\n评估结果 Top %d:\n%s", top_n, top_factors.to_string(index=False))

        return summary

    def compare_strategies(
        self,
        top_factors: pd.DataFrame,
        bt_start: str,
        bt_end: str,
        top_n: int = 5,
    ) -> List[BacktestResult]:
        """第2层：策略对比。"""
        logger.info("=" * 60)
        logger.info("第2层：策略对比")
        logger.info("=" * 60)

        factor_names = top_factors["factor_name"].head(top_n).tolist()
        factor_data = self._load_factor_data(bt_start, bt_end, factor_names)
        if factor_data.empty:
            logger.error("无因子数据")
            return []

        results = []

        for _, row in top_factors.head(top_n).iterrows():
            factor_name = row["factor_name"]
            ir = row["ir"]

            config = {
                "factor_name": factor_name,
                "ascending": ir < 0,
                "top_n": 30,
                "sma_short": 5,
                "sma_medium": 20,
                "buy_threshold": 0.6,
                "sell_threshold": 0.4,
                "reserve_cash_ratio": 0.1,
            }

            logger.info("测试因子: %s (IR=%.3f)", factor_name, ir)
            result = self._run_single_backtest(config, factor_data, bt_start, bt_end)
            if result:
                results.append(result)

        self._print_results(results, "策略对比结果")
        return results

    def grid_search_timing(
        self,
        best_factor: str,
        best_direction: bool,
        bt_start: str,
        bt_end: str,
    ) -> List[BacktestResult]:
        """第3层：择时参数网格搜索。"""
        logger.info("=" * 60)
        logger.info("第3层：择时参数网格搜索")
        logger.info("=" * 60)

        factor_data = self._load_factor_data(bt_start, bt_end, [best_factor])
        if factor_data.empty:
            logger.error("无因子数据")
            return []

        sma_short_options = [3, 5, 7]
        sma_medium_options = [10, 15, 20, 30]
        buy_threshold_options = [0.5, 0.55, 0.6, 0.65, 0.7]
        sell_threshold_options = [0.3, 0.35, 0.4, 0.45, 0.5]

        results = []
        total = (
            len(sma_short_options)
            * len(sma_medium_options)
            * len(buy_threshold_options)
            * len(sell_threshold_options)
        )
        count = 0

        for sma_s, sma_m, buy_t, sell_t in product(
            sma_short_options, sma_medium_options,
            buy_threshold_options, sell_threshold_options
        ):
            if sell_t >= buy_t:
                continue

            count += 1
            config = {
                "factor_name": best_factor,
                "ascending": best_direction,
                "top_n": 30,
                "sma_short": sma_s,
                "sma_medium": sma_m,
                "buy_threshold": buy_t,
                "sell_threshold": sell_t,
                "reserve_cash_ratio": 0.1,
            }

            logger.info("[%d/%d] 测试: sma=%d/%d, buy=%.2f, sell=%.2f",
                       count, total, sma_s, sma_m, buy_t, sell_t)
            result = self._run_single_backtest(config, factor_data, bt_start, bt_end)
            if result:
                results.append(result)

        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        self._print_results(results[:20], "择时参数 Top 20 (按夏普比率)")
        return results

    def walk_forward_validate(
        self,
        best_config: Dict,
        eval_periods: List[Tuple[str, str]],
        bt_periods: List[Tuple[str, str]],
    ) -> List[BacktestResult]:
        """第4层：滚动 Walk-forward 验证。"""
        logger.info("=" * 60)
        logger.info("第4层：滚动 Walk-forward 验证")
        logger.info("=" * 60)

        results = []
        for i, ((eval_s, eval_e), (bt_s, bt_e)) in enumerate(zip(eval_periods, bt_periods)):
            logger.info("滚动窗口 %d: 评估期=%s~%s, 回测期=%s~%s",
                       i+1, eval_s, eval_e, bt_s, bt_e)

            ev = FactorEvaluator(self.db)
            summary = ev.evaluate_all(
                factor_names=[best_config["factor_name"]],
                start_date=eval_s,
                end_date=eval_e,
                forward_days=5,
            )

            if summary.empty:
                logger.warning("评估期无数据，跳过")
                continue

            ir = summary.iloc[0]["ir"]
            config = best_config.copy()
            config["ascending"] = ir < 0

            factor_data = self._load_factor_data(bt_s, bt_e, [best_config["factor_name"]])
            if factor_data.empty:
                continue

            result = self._run_single_backtest(config, factor_data, bt_s, bt_e)
            if result:
                results.append(result)
                logger.info("  -> 总收益: %.2f%%, 夏普: %.2f",
                           result.total_return * 100, result.sharpe_ratio)

        if results:
            self._print_results(results, "Walk-forward 验证结果")
            avg_return = np.mean([r.total_return for r in results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results])
            logger.info("\n平均总收益: %.2f%%, 平均夏普: %.2f",
                       avg_return * 100, avg_sharpe)

        return results

    def _print_results(self, results: List[BacktestResult], title: str):
        """打印结果表格。"""
        if not results:
            logger.warning("无结果")
            return

        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        print(f"{'配置':<50} {'总收益':<10} {'年化':<10} {'最大回撤':<10} {'夏普':<8} {'交易数':<8}")
        print("-" * 100)

        for r in results:
            cfg_str = self._format_config(r.config)
            print(f"{cfg_str:<50} {r.total_return*100:>+8.2f}% {r.annualized_return*100:>+8.2f}% "
                  f"{r.max_drawdown*100:>+8.2f}% {r.sharpe_ratio:>+7.2f} {r.total_trades:<8d}")

        print("=" * 100 + "\n")

    def _format_config(self, config: Dict) -> str:
        """格式化配置为短字符串。"""
        return (f"{config['factor_name']}, "
                f"sma={config['sma_short']}/{config['sma_medium']}, "
                f"buy={config['buy_threshold']}, "
                f"sell={config['sell_threshold']}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["quick", "full", "factor_eval", "timing_grid"],
                   default="quick", help="优化模式")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--eval-start", default="2020-01-01")
    ap.add_argument("--eval-end", default="2020-06-30")
    ap.add_argument("--bt-start", default="2020-07-01")
    ap.add_argument("--bt-end", default="2024-12-31")
    ap.add_argument("--top-factors", type=int, default=20, help="评估因子数量")
    ap.add_argument("--forward-days", type=int, default=5)
    return ap.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cfg = load_config(args.config)
    db = Database(get_db_path(cfg))
    optimizer = Optimizer(db, cfg)

    if args.mode == "factor_eval":
        optimizer.evaluate_factors(
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            forward_days=args.forward_days,
            top_n=args.top_factors,
        )

    elif args.mode == "quick":
        summary = optimizer.evaluate_factors(
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            forward_days=args.forward_days,
            top_n=args.top_factors,
        )

        if not summary.empty:
            best_factor = summary.iloc[0]["factor_name"]
            best_ir = summary.iloc[0]["ir"]
            best_direction = best_ir < 0

            logger.info("\n最优因子: %s (IR=%.3f)", best_factor, best_ir)

            optimizer.compare_strategies(
                top_factors=summary,
                bt_start=args.bt_start,
                bt_end=args.bt_end,
                top_n=5,
            )

    elif args.mode == "full":
        summary = optimizer.evaluate_factors(
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            forward_days=args.forward_days,
            top_n=args.top_factors,
        )

        if not summary.empty:
            best_factor = summary.iloc[0]["factor_name"]
            best_ir = summary.iloc[0]["ir"]
            best_direction = best_ir < 0

            strategy_results = optimizer.compare_strategies(
                top_factors=summary,
                bt_start=args.bt_start,
                bt_end=args.bt_end,
                top_n=5,
            )

            if strategy_results:
                best_config = max(strategy_results, key=lambda x: x.sharpe_ratio).config
                optimizer.grid_search_timing(
                    best_factor=best_config["factor_name"],
                    best_direction=best_config["ascending"],
                    bt_start=args.bt_start,
                    bt_end=args.bt_end,
                )

    elif args.mode == "timing_grid":
        summary = optimizer.evaluate_factors(
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            forward_days=args.forward_days,
            top_n=args.top_factors,
        )

        if not summary.empty:
            best_factor = summary.iloc[0]["factor_name"]
            best_ir = summary.iloc[0]["ir"]
            best_direction = best_ir < 0

            optimizer.grid_search_timing(
                best_factor=best_factor,
                best_direction=best_direction,
                bt_start=args.bt_start,
                bt_end=args.bt_end,
            )

    db.close()
    logger.info("优化完成!")


if __name__ == "__main__":
    main()
