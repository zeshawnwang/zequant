#!/usr/bin/env python3
"""对比回测脚本：原配置 vs 新配置。

用法
----
python3 scripts/run_backtest_comparison.py \
    --eval-start 2020-01-01 --eval-end 2020-06-30 \
    --bt-start   2020-07-01 --bt-end   2024-12-31
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.config import load_config, get_db_path
from core.database import Database
from core.backtest import BacktestEngine
from core.universe import SymbolUniverse, UniverseConfig
from core.strategy_hub import create as create_strategy, get_meta as get_strategy_meta
from core.factor_evaluator import FactorEvaluator
import strategies  # noqa: F401
import factors      # noqa: F401

logger = logging.getLogger("zequant.comparison")

# 原配置（30 只持仓）
ORIGINAL_CONFIG = {
    "top_n": 30,
    "top_factors": 8,
    "timing": {
        "sma_short": 5,
        "sma_medium": 20,
        "buy_threshold": 0.55,
        "sell_threshold": 0.4,
    },
    "portfolio": {
        "type": "equal_weight",
        "reserve_cash_ratio": 0.1,
    },
}

# 新配置（15 只持仓，精简版）
NEW_CONFIG = {
    "top_n": 15,
    "top_factors": 5,
    "timing": {
        "sma_short": 3,
        "sma_medium": 10,
        "buy_threshold": 0.65,
        "sell_threshold": 0.45,
    },
    "portfolio": {
        "type": "equal_weight",
        "reserve_cash_ratio": 0.15,
    },
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--eval-start", required=True, help="评估期开始")
    ap.add_argument("--eval-end", required=True, help="评估期结束")
    ap.add_argument("--bt-start", required=True, help="回测开始")
    ap.add_argument("--bt-end", required=True, help="回测结束")
    ap.add_argument("--forward-days", type=int, default=None, help="前瞻收益窗口")
    return ap.parse_args()


def _run_evaluation(db: Database, eval_start: str, eval_end: str, forward_days: int) -> pd.DataFrame:
    """跑因子评估，仅评估 alpha 因子（a* 开头）。"""
    ev = FactorEvaluator(db)
    all_names = db.list_factor_columns()
    alpha_names = sorted(
        [n for n in all_names if n.startswith("a") and n[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    if not alpha_names:
        raise RuntimeError("没有可评估的 alpha 因子，请先跑 compute_alpha101_full")
    return ev.evaluate_all(
        factor_names=alpha_names,
        start_date=eval_start,
        end_date=eval_end,
        forward_days=forward_days,
    )


def _run_backtest(db, eval_summary, strategy_config, bt_start, bt_end, cfg):
    """运行单次回测，返回 BacktestReport。"""
    # 创建策略
    strategy = create_strategy(
        "alpha101_walk_forward",
        db=db,
        eval_summary=eval_summary,
        strategy_config=strategy_config,
    )
    logger.info("创建策略: %s", strategy.name)
    logger.info("配置: top_n=%d, top_factors=%d", strategy_config["top_n"], strategy_config["top_factors"])

    # Universe 与回测引擎
    universe_cfg = UniverseConfig.from_config(cfg.get("universe", {}))
    universe = SymbolUniverse(db, universe_cfg)
    initial_capital = cfg["backtest"]["initial_capital"]
    engine = BacktestEngine(
        initial_capital=initial_capital,
        fee_config=cfg.get("fees", {}),
        risk_config=cfg.get("risk", {}),
        universe=universe,
    )

    # 加载因子数据
    meta = get_strategy_meta("alpha101_walk_forward")
    available = set(db.list_factor_columns())
    wanted = set(meta.timing_factors) | set(strategy.selector.factor_names)
    factor_cols = sorted([c for c in wanted if c in available])
    factor_data = db.get_factors(
        start_date=bt_start,
        end_date=bt_end,
        factor_names=factor_cols if factor_cols else None,
        with_close=True,
    )
    if factor_data.empty:
        raise SystemExit("无回测期因子数据")

    # 回测引擎需要 open 列来执行待成交订单,补充进去
    bars = db.get_daily_bars(start_date=bt_start, end_date=bt_end)
    if not bars.empty and "open" in bars.columns:
        open_data = bars[["date", "symbol", "open"]].copy()
        open_data["date"] = pd.to_datetime(open_data["date"])
        factor_data["date"] = pd.to_datetime(factor_data["date"])
        factor_data = factor_data.merge(open_data, on=["date", "symbol"], how="left")

    logger.info("因子数据: %s 条, 列=%d", f"{len(factor_data):,}", len(factor_data.columns) - 7)

    # 运行回测
    report = engine.run(
        strategy=strategy,
        factor_data=factor_data,
        start_date=bt_start,
        end_date=bt_end,
        rebalance_freq=cfg["backtest"].get("rebalance_freq", "1d"),
    )
    return report


def _print_comparison(original: "BacktestReport", new: "BacktestReport"):
    """打印对比报告。"""
    print("\n" + "=" * 80)
    print("策略配置对比")
    print("=" * 80)
    print()
    print("配置参数:")
    print(f"  {'参数':<20} {'原配置 (30只)':<20} {'新配置 (15只)':<20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    print(f"  {'持仓数量':<18} {30:<20} {15:<20}")
    print(f"  {'因子数量':<18} {8:<20} {5:<20}")
    print(f"  {'短期均线':<18} {5:<20} {3:<20}")
    print(f"  {'中期均线':<18} {20:<20} {10:<20}")
    print(f"  {'买入阈值':<18} {0.55:<20.2f} {0.65:<20.2f}")
    print(f"  {'卖出阈值':<18} {0.40:<20.2f} {0.45:<20.2f}")
    print(f"  {'预留现金':<18} {'10%':<20} {'15%':<20}")
    print()
    print("回测结果:")
    print(f"  {'指标':<20} {'原配置':<20} {'新配置':<20} {'变化':<15}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*15}")
    print(f"  {'初始本金':<18} {original.initial_capital:<20,.0f} {new.initial_capital:<20,.0f} {'-':<15}")
    print(f"  {'期末总值':<18} {original.final_value:<20,.2f} {new.final_value:<20,.2f} {new.final_value - original.final_value:>+,.2f}")
    print(f"  {'绝对盈亏':<18} {original.final_value - original.initial_capital:<+20,.2f} {new.final_value - new.initial_capital:<+20,.2f} {(new.final_value - new.initial_capital) - (original.final_value - original.initial_capital):>+,.2f}")
    print()
    print(f"  {'总收益率':<18} {original.total_return*100:<+19.2f}% {new.total_return*100:<+19.2f}% {new.total_return - original.total_return:>+,.2f}%")
    print(f"  {'年化收益':<18} {original.annualized_return*100:<+19.2f}% {new.annualized_return*100:<+19.2f}% {new.annualized_return - original.annualized_return:>+,.2f}%")
    print(f"  {'最大回撤':<18} {original.max_drawdown*100:<+19.2f}% {new.max_drawdown*100:<+19.2f}% {new.max_drawdown - original.max_drawdown:>+,.2f}%")
    print(f"  {'夏普比率':<18} {original.sharpe_ratio:<20.2f} {new.sharpe_ratio:<20.2f} {new.sharpe_ratio - original.sharpe_ratio:>+,.2f}")
    print(f"  {'胜率':<18} {original.win_rate*100:<19.2f}% {new.win_rate*100:<19.2f}% {new.win_rate - original.win_rate:>+,.2f}%")
    print(f"  {'盈亏比':<18} {original.profit_factor:<20.2f} {new.profit_factor:<20.2f} {new.profit_factor - original.profit_factor:>+,.2f}")
    print(f"  {'交易次数':<18} {original.total_trades:<20d} {new.total_trades:<20d} {new.total_trades - original.total_trades:>+d}")
    print()
    print(f"  {'期末现金':<18} {original.final_cash:<20,.2f} {new.final_cash:<20,.2f} {new.final_cash - original.final_cash:>+,.2f}")
    print(f"  {'期末持仓':<18} {original.final_position_value:<20,.2f} {new.final_position_value:<20,.2f} {new.final_position_value - original.final_position_value:>+,.2f}")
    print(f"  {'末日持仓数':<16} {len(original.final_positions):<20d} {len(new.final_positions):<20d} {len(new.final_positions) - len(original.final_positions):>+d}")
    print()
    print("=" * 80)


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    forward_days = args.forward_days if args.forward_days is not None else int(cfg["factors"]["forward_days"])
    db = Database(get_db_path(cfg))

    # 1) 因子评估
    logger.info("评估期: %s ~ %s, 前瞻 %d 日", args.eval_start, args.eval_end, forward_days)
    eval_summary = _run_evaluation(db, args.eval_start, args.eval_end, forward_days)
    logger.info("评估摘要(|IR| 前 5):\n%s", eval_summary.head(5).to_string(index=False))

    # 2) 原配置回测
    logger.info("=" * 60)
    logger.info("开始原配置回测 (30只持仓)")
    logger.info("=" * 60)
    original_report = _run_backtest(
        db, eval_summary, ORIGINAL_CONFIG,
        args.bt_start, args.bt_end, cfg,
    )

    # 3) 新配置回测
    logger.info("=" * 60)
    logger.info("开始新配置回测 (15只持仓)")
    logger.info("=" * 60)
    new_report = _run_backtest(
        db, eval_summary, NEW_CONFIG,
        args.bt_start, args.bt_end, cfg,
    )

    # 4) 对比
    _print_comparison(original_report, new_report)

    db.close()


if __name__ == "__main__":
    main()
