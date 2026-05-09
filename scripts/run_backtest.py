#!/usr/bin/env python3
"""统一回测入口 —— 静态策略 与 评估驱动策略 共用一套命令行。

用法
----
1) 静态策略(默认):
   python3 scripts/run_backtest.py --strategy momentum_top50
   python3 scripts/run_backtest.py --strategy low_vol_top50 --top-n 30

2) 评估驱动策略(StrategyHub.requires_evaluation=True):
   入口会自动:① 在 [eval-start, eval-end] 跑 FactorEvaluator
              ② 把 summary 注入 factory
              ③ 在 [bt-start, bt-end] 样本外回测
   python3 scripts/run_backtest.py --strategy alpha101_walk_forward \\
       --eval-start 2024-01-01 --eval-end 2024-06-30 \\
       --bt-start   2024-07-01 --bt-end   2024-12-31 \\
       --top-factors 8 --top-n 30

3) 列出所有已注册策略:
   python3 scripts/run_backtest.py --list
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

from core.config import load_config, get_db_path, get_strategy_config
from core.database import Database
from core.backtest import BacktestEngine
from core.universe import SymbolUniverse, UniverseConfig
from core.strategy_hub import StrategyHub
from core.factor_evaluator import FactorEvaluator
import strategies  # noqa: F401  触发策略注册
import factors      # noqa: F401  触发因子注册(便于按需重算)

logger = logging.getLogger("zequant.backtest")


# ===== 命令行 =============================================================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--strategy", default="momentum_top50",
                    help="StrategyHub 中注册的策略名,--list 可查看所有")
    ap.add_argument("--top-n", type=int, default=50, help="选股数量")
    ap.add_argument("--initial-capital", type=float, default=None,
                    help="覆盖 config 的初始资金")
    ap.add_argument("--bt-start", default=None,
                    help="回测开始日(默认读 config.backtest.start_date)")
    ap.add_argument("--bt-end", default=None,
                    help="回测结束日(默认读 config.backtest.end_date)")

    # 评估驱动策略专用 —— 只对 requires_evaluation=True 的策略生效
    ap.add_argument("--eval-start", default=None,
                    help="评估期开始,仅 requires_evaluation 策略需要")
    ap.add_argument("--eval-end", default=None,
                    help="评估期结束,仅 requires_evaluation 策略需要")
    ap.add_argument("--top-factors", type=int, default=8,
                    help="评估驱动策略:取 |IR| 前 N 个因子")
    ap.add_argument("--forward-days", type=int, default=None,
                    help="评估期前瞻收益窗口(默认读 config.factors.forward_days)")
    ap.add_argument("--min-abs-ir", type=float, default=0.2,
                    help="from_registry 类策略的 |IR| 下限,会注入 factory 的 min_abs_ir kwarg")

    ap.add_argument("--list", action="store_true",
                    help="列出所有已注册策略并退出")
    return ap.parse_args()


# ===== 主流程 =============================================================

def _list_strategies() -> None:
    print("已注册策略:")
    for name in StrategyHub.list_all():
        print(f"  - {StrategyHub.describe(name)}")


def _select_eval_factor_names(db: Database, eval_factor_filter: str) -> list:
    """根据策略元数据中的 eval_factor_filter 选择评估候选因子。

    目前实现:
      - "alpha" -> 仅 a1, a2, ... 系列(库中以小写 a + 数字开头的列)
      - None / 其他 -> 库中全部因子列
    """
    all_names = db.list_factor_columns()
    if eval_factor_filter == "alpha":
        return sorted(
            [n for n in all_names if n.startswith("a") and n[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
    return all_names


def _load_factors_for_strategy(
    db: Database,
    strategy_name: str,
    bt_start: str,
    bt_end: str,
    extra_factors: list,
) -> pd.DataFrame:
    """按策略元信息加载因子宽表(含择时器依赖列)。"""
    meta = StrategyHub.get_meta(strategy_name)
    available = set(db.list_factor_columns())
    wanted = set(extra_factors) | set(meta.timing_factors)
    factor_cols = sorted([c for c in wanted if c in available])
    missing = [c for c in (set(meta.timing_factors) - set(factor_cols))]
    if missing:
        logger.warning("策略 %s 声明但库中缺失的择时因子: %s", strategy_name, missing)

    return db.get_factors(
        start_date=bt_start, end_date=bt_end,
        factor_names=factor_cols if factor_cols else None,
        with_close=True,
    )


def _run_evaluation(
    db: Database,
    eval_start: str,
    eval_end: str,
    forward_days: int,
    factor_filter: str = None,
) -> pd.DataFrame:
    """跑因子评估,返回 summary。factor_filter='alpha' 时仅评估 a* 因子。"""
    ev = FactorEvaluator(db)
    names = _select_eval_factor_names(db, factor_filter)
    if not names:
        raise RuntimeError("没有可评估的因子,先跑 compute_factors / compute_alpha101_full")
    return ev.evaluate_all(
        factor_names=names,
        start_date=eval_start, end_date=eval_end,
        forward_days=forward_days,
    )


def main() -> None:
    args = _parse_args()
    if args.list:
        _list_strategies()
        return

    # 1) 配置
    cfg = load_config(args.config)
    bt_start = args.bt_start or cfg["backtest"]["start_date"]
    bt_end = args.bt_end or cfg["backtest"]["end_date"]
    initial_capital = args.initial_capital or cfg["backtest"]["initial_capital"]
    forward_days = args.forward_days if args.forward_days is not None \
        else int(cfg["factors"]["forward_days"])

    db = Database(get_db_path(cfg))
    meta = StrategyHub.get_meta(args.strategy)

    # 2) Universe 与回测引擎
    universe_cfg = UniverseConfig.from_config(cfg.get("universe", {}))
    logger.info("Universe 配置: %s", universe_cfg)
    universe = SymbolUniverse(db, universe_cfg)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        fee_config=cfg.get("fees", {}),
        risk_config=cfg.get("risk", {}),
        universe=universe,
    )

    # 3) 读取策略专属配置并组装 factory 参数
    strategy_cfg = get_strategy_config(cfg, args.strategy)
    factory_kwargs = {
        "db": db,
        "top_n": args.top_n,
        "min_abs_ir": args.min_abs_ir,
        "strategy_config": strategy_cfg,
    }

    # 4) 因子评估(若策略需要)
    if meta.requires_evaluation:
        if not args.eval_start or not args.eval_end:
            raise SystemExit(
                f"策略 {args.strategy} 需要因子评估,请提供 --eval-start / --eval-end"
            )
        logger.info("评估期: %s ~ %s,前瞻 %d 日", args.eval_start, args.eval_end, forward_days)
        eval_summary = _run_evaluation(
            db, args.eval_start, args.eval_end, forward_days,
            factor_filter=meta.eval_factor_filter,
        )
        factory_kwargs["eval_summary"] = eval_summary
        factory_kwargs["top_factors"] = args.top_factors
        logger.info("\n评估摘要(|IR| 前 5):\n%s",
                    eval_summary.head(5).to_string(index=False))

    # 5) 创建策略
    strategy = StrategyHub.create(args.strategy, **factory_kwargs)
    logger.info("策略: %s  (注册名: %s)", strategy.name, args.strategy)
    logger.info(strategy.get_description())

    # 6) 加载因子(含择时所需)
    selector_factors = list(getattr(strategy.selector, "factor_names", [])) \
        or [getattr(strategy.selector, "factor_name", None)]
    selector_factors = [f for f in selector_factors if f]
    factor_data = _load_factors_for_strategy(
        db, args.strategy, bt_start, bt_end, selector_factors,
    )
    if factor_data.empty:
        raise SystemExit(
            f"无回测期因子数据 ({bt_start}~{bt_end}),"
            "请先 compute_factors / compute_alpha101_full"
        )
    logger.info(
        "因子数据: %s 条, %s ~ %s,列=%d",
        f"{len(factor_data):,}",
        factor_data["date"].min().strftime("%Y-%m-%d"),
        factor_data["date"].max().strftime("%Y-%m-%d"),
        sum(c not in ("date", "symbol", "close", "pct_change", "volume", "amount")
            for c in factor_data.columns),
    )

    # 7) 回测
    logger.info("运行回测: %s ~ %s ...", bt_start, bt_end)
    report = engine.run(
        strategy=strategy,
        factor_data=factor_data,
        start_date=bt_start, end_date=bt_end,
        rebalance_freq=cfg["backtest"].get("rebalance_freq", "1d"),
    )

    # 8) 报告
    print("\n" + report.pretty_print(top_positions=20, top_selections=3))

    db.close()


if __name__ == "__main__":
    main()