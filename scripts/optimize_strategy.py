#!/usr/bin/env python3
"""
策略优化完整流程脚本。

执行步骤：
1. 因子筛选：回测所有因子，选出30个最佳因子
2. 权重优化：用遗传算法优化5个最佳权重配置
3. 归因分析：分析最佳配置的收益来源
4. 保存配置：将结果保存到YAML配置文件

使用方法：
    python scripts/optimize_strategy.py \
        --factor_count 30 \
        --weight_configs 5 \
        --top_n 30 \
        --pop_size 50 \
        --generations 100
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import Database
from core.optimization.risk_constraints import RiskConstraints
from core.optimization.factor_selector import FactorSelector
from core.optimization.weight_optimizer import GeneticWeightOptimizer
from core.optimization.attribution import StrategyAttribution
from core.optimization.config_manager import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="策略优化完整流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--factor_count", type=int, default=30, help="筛选多少个因子")
    parser.add_argument("--weight_configs", type=int, default=5, help="优化出多少个权重配置")
    parser.add_argument("--top_n", type=int, default=30, help="选股每次选多少只")
    parser.add_argument("--pop_size", type=int, default=50, help="遗传算法种群大小")
    parser.add_argument("--generations", type=int, default=100, help="遗传算法迭代代数")
    parser.add_argument("--skip_factor", action="store_true", help="跳过因子筛选，从已有配置加载因子")

    args = parser.parse_args()

    # =============================================
    # 0. 初始化
    # =============================================
    logger.info("=" * 80)
    logger.info("策略优化完整流程开始")
    logger.info("=" * 80)

    db = Database()
    config_manager = ConfigManager()

    # 风险约束设置
    risk_constraints = RiskConstraints(
        max_drawdown=0.40,
        single_stock_weight=0.15,
        single_sector_weight=0.25,
        max_volatility=0.30,
        max_turnover=1.00,
        min_calmar_ratio=0.5,
        min_win_rate=0.50,
    )

    # 时间范围设置
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2020-12-31"
    VAL_START = "2021-01-01"
    VAL_END = "2023-12-31"
    OOS_START = "2024-01-01"
    OOS_END = "2026-04-30"

    logger.info(f"因子筛选期: {TRAIN_START} ~ {TRAIN_END}")
    logger.info(f"权重优化期: {VAL_START} ~ {VAL_END}")
    logger.info(f"样本外测试: {OOS_START} ~ {OOS_END}")

    # =============================================
    # 1. 第一阶段：因子筛选
    # =============================================
    logger.info("\n" + "=" * 80)
    logger.info("第一阶段：因子筛选")
    logger.info("=" * 80)

    if args.skip_factor:
        logger.info("跳过因子筛选，从已有配置加载...")
        existing_config = config_manager.load()
        if existing_config is None:
            logger.warning("没有找到现有配置，继续执行因子筛选...")
            args.skip_factor = False
        else:
            selected_factor_names = [f.name for f in existing_config.best_factors]
            logger.info(f"从配置加载了 {len(selected_factor_names)} 个因子")

    if not args.skip_factor:
        all_factors = db.list_factor_columns()
        logger.info(f"共有 {len(all_factors)} 个因子可供筛选")

        # 先做一个快速筛选（用前100个因子做演示，避免时间太长）
        factor_selector = FactorSelector(
            db=db,
            risk_constraints=risk_constraints,
            top_n=args.top_n,
            target_factor_count=args.factor_count,
        )

        # 为了演示速度，先只选前100个因子
        demo_factors = all_factors[:100] if len(all_factors) > 100 else all_factors
        logger.info(f"为演示速度，先只用 {len(demo_factors)} 个因子进行筛选...")

        factor_results = factor_selector.run(
            factor_names=demo_factors,
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            parallel=False,
        )

        top_factor_results = factor_selector.get_top_factors()
        logger.info(f"筛选完成，选出 {len(top_factor_results)} 个最佳因子")

        selected_factor_names = [fr.factor_name for fr in top_factor_results]

        df_results = factor_selector.get_results_df()
        if not df_results.empty:
            logger.info("\n因子筛选结果前10名：")
            print(df_results.head(10).to_string(index=False))
    else:
        top_factor_results = []

    # =============================================
    # 2. 第二阶段：权重优化
    # =============================================
    logger.info("\n" + "=" * 80)
    logger.info("第二阶段：权重优化")
    logger.info("=" * 80)

    if len(selected_factor_names) < 3:
        logger.error("有效因子数量不足，无法进行权重优化")
        return

    weight_optimizer = GeneticWeightOptimizer(
        db=db,
        risk_constraints=risk_constraints,
        factor_names=selected_factor_names,
        top_n=args.top_n,
        population_size=args.pop_size,
        generations=args.generations,
    )

    best_weights = weight_optimizer.run(
        start_date=VAL_START,
        end_date=VAL_END,
        target_config_count=args.weight_configs,
    )

    logger.info(f"权重优化完成，获得 {len(best_weights)} 个有效配置")

    df_weights = weight_optimizer.get_results_df()
    if not df_weights.empty:
        logger.info("\n权重优化结果：")
        print(df_weights.to_string(index=False))

    # =============================================
    # 3. 第三阶段：归因分析
    # =============================================
    logger.info("\n" + "=" * 80)
    logger.info("第三阶段：归因分析")
    logger.info("=" * 80)

    if best_weights:
        best_config = best_weights[0]
        logger.info(f"对最佳配置 {best_config.name} 进行归因分析...")

        attribution = StrategyAttribution(db=db)
        attr_result = attribution.analyze(
            selected_factors=selected_factor_names,
            weights=best_config.weights,
            start_date=VAL_START,
            end_date=VAL_END,
        )
        attr_result.pretty_print()

    # =============================================
    # 4. 保存配置
    # =============================================
    logger.info("\n" + "=" * 80)
    logger.info("第四阶段：保存配置")
    logger.info("=" * 80)

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if top_factor_results and best_weights:
        config = config_manager.build_config(
            factor_results=top_factor_results,
            weight_configs=best_weights,
            risk_constraints=risk_constraints,
            created_at=created_at,
        )
        config_manager.save(config)

    # =============================================
    # 结束
    # =============================================
    logger.info("\n" + "=" * 80)
    logger.info("策略优化完整流程结束")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
