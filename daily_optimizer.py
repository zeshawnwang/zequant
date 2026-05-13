"""分类因子优化框架 - 支持断点续传。

按10个因子类别分组优化：
1. 阶段1：每类因子单因子回测，取最佳5个 → 50个因子
2. 阶段2：50因子多因子+遗传算法权重优化，取5个配比

所有中间结果保存到 daily/YYYY-MM-DD/ 目录下，支持断点续传。
"""
from __future__ import annotations
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.database import Database
from core.execution import BacktestEngine
from core.screening.impl.factor_rank import FactorRankSelector
from core.screening.impl.multi_factor import MultiFactorSelector
from core.timings.impl.trend_volatility import TrendVolatilityTiming
from core.strategies.base.strategy import SignalStrategy
from core.signals.base.composer import LayeredComposer
from core.optimization.base.factor_categories import FACTOR_CATEGORIES, get_db_factors_by_category
from core.optimization.base.risk_constraints import RiskConstraints


@dataclass
class BacktestRecord:
    """单次回测记录。"""
    timestamp: str
    factor_name: str
    category: str
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    win_rate: float
    risk_passed: bool
    score: float


@dataclass
class OptimizationCheckpoint:
    """优化检查点 - 用于断点续传。"""
    timestamp: str
    stage: str
    category: Optional[str] = None
    current_factor_index: int = 0
    completed_categories: List[str] = field(default_factory=list)
    selected_factors: Dict[str, List[str]] = field(default_factory=dict)
    best_weights: List[Dict] = field(default_factory=list)
    backtest_records: List[Dict] = field(default_factory=list)

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': self.timestamp,
                'stage': self.stage,
                'category': self.category,
                'current_factor_index': self.current_factor_index,
                'completed_categories': self.completed_categories,
                'selected_factors': self.selected_factors,
                'best_weights': self.best_weights,
                'backtest_records': self.backtest_records
            }, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Path) -> 'OptimizationCheckpoint':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return OptimizationCheckpoint(**data)


class DailyOptimizer:
    """分类因子优化器 - 支持断点续传。"""

    def __init__(
        self,
        db_path: str = "./data/quant_data.db",
        output_dir: str = "./daily",
        top_n_per_category: int = 5,
        max_drawdown: float = 0.60,
    ):
        self.db = Database(db_path)
        self.output_dir = Path(output_dir)
        self.today_dir = self.output_dir / datetime.now().strftime('%Y-%m-%d')
        self.today_dir.mkdir(parents=True, exist_ok=True)

        self.stage1_constraints = RiskConstraints(
            max_drawdown=0.80,
            single_stock_weight=0.30,
            single_sector_weight=0.50,
            max_volatility=0.60,
            max_turnover=2.0,
            min_calmar_ratio=0.0,
            min_win_rate=0.0,
        )
        self.risk_constraints = RiskConstraints(
            max_drawdown=max_drawdown,
            single_stock_weight=0.15,
            single_sector_weight=0.25,
            max_volatility=0.30,
            max_turnover=1.0,
            min_calmar_ratio=0.5,
            min_win_rate=0.40,
        )

        self.top_n_per_category = top_n_per_category
        self.checkpoint_file = self.today_dir / "checkpoint.json"
        self.log_file = self.today_dir / "optimization.log"

        self._setup_logging()

    def _setup_logging(self):
        """配置日志系统。"""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_required_factors(self) -> List[str]:
        return ['momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal', 'volatility_20', 'volume_ratio', 'boll_position']

    def _backtest_factor(
        self,
        factor_name: str,
        category: str,
        start_date: str,
        end_date: str,
    ) -> Optional[BacktestRecord]:
        """单因子回测（向量化加速版）。"""
        try:
            required = self._get_required_factors()
            all_req = list(set(required + [factor_name]))
            factor_data = self.db.get_factors(
                factor_names=all_req,
                start_date=start_date,
                end_date=end_date,
                with_close=True,
            )
            if factor_data is None or factor_data.empty:
                self.logger.warning(f"因子 {factor_name} 没有数据")
                return None

            factor_data = factor_data.sort_values("date")
            val_cols = [c for c in factor_data.columns if c not in ("date", "symbol")]
            pivots = {}
            for col in val_cols:
                pivots[col] = factor_data.pivot_table(
                    index="date", columns="symbol", values=col
                ).ffill()

            dates = list(pivots[factor_name].index)
            if len(dates) < 50:
                return None

            for col in list(pivots.keys()):
                if pivots[col].shape[0] != len(dates):
                    pivots[col] = pivots[col].reindex(dates).ffill()

            top_n = self.top_n_per_category
            cash = 1_000_000.0
            positions: Dict[str, int] = {}
            daily_values: List[float] = [cash]
            rebalance_counter = 0
            n = len(dates)

            for di in range(n):
                if di < 20:
                    daily_values.append(cash)
                    continue

                factor_series = pivots[factor_name].iloc[di].dropna()
                if factor_series.empty:
                    daily_values.append(daily_values[-1])
                    continue

                ranked = factor_series.sort_values(ascending=False)
                candidates = ranked.head(top_n * 3).index.tolist()

                selected = []
                for sym in candidates:
                    macd = pivots.get("macd", pd.DataFrame()).iloc[di].get(sym, np.nan)
                    macd_sig = pivots.get("macd_signal", pd.DataFrame()).iloc[di].get(sym, np.nan)
                    m5 = pivots.get("momentum_5", pd.DataFrame()).iloc[di].get(sym, np.nan)
                    m20 = pivots.get("momentum_20", pd.DataFrame()).iloc[di].get(sym, np.nan)
                    rsi = pivots.get("rsi_14", pd.DataFrame()).iloc[di].get(sym, np.nan)
                    vol = pivots.get("volatility_20", pd.DataFrame()).iloc[di].get(sym, np.nan)

                    if pd.notna(vol) and vol > 0.05:
                        continue

                    trend_scores = []
                    if pd.notna(macd) and pd.notna(macd_sig):
                        trend_scores.append(1.0 if macd > macd_sig else 0.0)
                    if pd.notna(m5) and pd.notna(m20):
                        if m5 > 0 and m5 > m20:
                            trend_scores.append(1.0)
                        elif m5 < 0:
                            trend_scores.append(0.0)
                        else:
                            trend_scores.append(0.5)
                    if pd.notna(rsi):
                        if 50 <= rsi <= 70:
                            trend_scores.append(1.0)
                        elif 30 <= rsi < 50:
                            trend_scores.append(0.5)
                        else:
                            trend_scores.append(0.0)

                    trend_score = float(np.mean(trend_scores)) if trend_scores else 0.5
                    if trend_score >= 0.6:
                        selected.append(sym)

                if not selected:
                    daily_values.append(daily_values[-1])
                    rebalance_counter += 1
                    continue

                if rebalance_counter % 21 == 0:
                    today_close = pivots["close"].iloc[di]
                    for sym, shares in positions.items():
                        px = today_close.get(sym, np.nan)
                        if pd.notna(px) and px > 0:
                            cash += shares * float(px)
                    positions = {}

                    target_weight = 1.0 / max(len(selected), 1)
                    for sym in selected:
                        px = today_close.get(sym, np.nan)
                        if pd.notna(px) and px > 0:
                            target_value = cash * target_weight
                            shares = int(target_value / float(px) / 100) * 100
                            if shares >= 100:
                                cost = shares * float(px)
                                if cost <= cash:
                                    cash -= cost
                                    positions[sym] = positions.get(sym, 0) + shares

                today_close = pivots["close"].iloc[di]
                total = cash
                for sym, shares in positions.items():
                    px = today_close.get(sym, np.nan)
                    if pd.notna(px) and px > 0:
                        total += shares * float(px)
                    else:
                        total += shares * float(pivots["close"].iloc[di - 1].get(sym, 0))
                daily_values.append(total)
                rebalance_counter += 1

            if len(daily_values) < 2 or daily_values[-1] <= 0:
                return None

            total_return = daily_values[-1] / daily_values[0] - 1
            returns = np.array([
                daily_values[i] / daily_values[i - 1] - 1
                for i in range(1, len(daily_values))
                if daily_values[i - 1] > 0
            ])
            if len(returns) == 0:
                return None

            ann_factor = 252 / max(len(returns), 1)
            annualized_return = (1 + total_return) ** min(ann_factor, 10) - 1

            peak = np.maximum.accumulate(daily_values)
            dd = (np.array(daily_values) - peak) / peak
            max_drawdown = float(np.min(dd))

            if np.std(returns) > 0:
                sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            else:
                sharpe_ratio = 0.0

            win_rate = float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0.0
            calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

            risk_result = self.stage1_constraints.check_backtest_result(
                annual_return=annualized_return,
                max_drawdown=abs(max_drawdown),
                volatility=float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.5,
                calmar_ratio=calmar,
                win_rate=win_rate,
                turnover=None,
            )

            score = -1
            if risk_result.passed:
                score = (annualized_return * 0.3 + sharpe_ratio * 0.25 +
                        calmar * 0.3 + win_rate * 0.15)

            return BacktestRecord(
                timestamp=datetime.now().isoformat(),
                factor_name=factor_name,
                category=category,
                total_return=total_return,
                annual_return=annualized_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar,
                win_rate=win_rate,
                risk_passed=risk_result.passed,
                score=score,
            )

        except Exception as e:
            self.logger.error(f"因子 {factor_name} 回测异常: {e}")
            return None

    def _save_records(self, records: List[BacktestRecord]):
        records_file = self.today_dir / "backtest_records.json"
        data = [asdict(r) for r in records]
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> Optional[OptimizationCheckpoint]:
        if self.checkpoint_file.exists():
            try:
                return OptimizationCheckpoint.load(self.checkpoint_file)
            except Exception as e:
                self.logger.warning(f"加载检查点失败: {e}")
        return None

    def _save_checkpoint(self, checkpoint: OptimizationCheckpoint):
        checkpoint.timestamp = datetime.now().isoformat()
        checkpoint.save(self.checkpoint_file)

    def run_stage1(self, start_date: str, end_date: str) -> Dict[str, List[str]]:
        """阶段1：每类因子单因子回测，取最佳5个。"""
        self.logger.info("=" * 60)
        self.logger.info("阶段1：分类因子单因子回测")
        self.logger.info("=" * 60)

        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.stage == "stage1":
            self.logger.info(f"从检查点恢复: 已完成 {len(checkpoint.backtest_records)} 次回测")
            completed_categories = set(checkpoint.completed_categories)
        else:
            checkpoint = OptimizationCheckpoint(timestamp=datetime.now().isoformat(), stage="stage1")
            completed_categories = set()

        db_factors = get_db_factors_by_category(self.db)
        all_records = list(checkpoint.backtest_records)

        for category, factors in db_factors.items():
            if category in completed_categories:
                self.logger.info(f"跳过已完成的类别: {category}")
                continue

            self.logger.info(f"\n处理类别: {category} ({len(factors)} 个因子)")
            checkpoint.category = category

            for i, factor in enumerate(factors, 1):
                if any(r['factor_name'] == factor for r in all_records):
                    self.logger.info(f"[{i}/{len(factors)}] 跳过已有: {factor}")
                    continue

                self.logger.info(f"[{i}/{len(factors)}] 回测因子: {factor}")
                record = self._backtest_factor(factor, category, start_date, end_date)

                if record:
                    all_records.append(asdict(record))
                    self._save_records([BacktestRecord(**r) for r in all_records])

                    status = "✓" if record.risk_passed else "✗"
                    self.logger.info(f"  {status} 年化: {record.annual_return:+.2%} "
                                   f"回撤: {abs(record.max_drawdown):.2%} 得分: {record.score:.4f}")

                checkpoint.current_factor_index = i

                if i % 5 == 0:
                    self._save_checkpoint(checkpoint)

            completed_categories.add(category)
            checkpoint.completed_categories = list(completed_categories)
            checkpoint.current_factor_index = 0
            self._save_checkpoint(checkpoint)

        checkpoint.selected_factors = self._select_top_factors(all_records)
        self._save_checkpoint(checkpoint)

        for cat, factors in checkpoint.selected_factors.items():
            self.logger.info(f"类别 {cat} 选出: {factors}")

        return checkpoint.selected_factors

    def _select_top_factors(self, records: List[Dict]) -> Dict[str, List[str]]:
        """选择 TOP 50 因子,保证每个类别至少1个,共7类保底7个,剩余43个按全局得分择优补充。"""
        TOTAL_FACTORS = 50
        GUARANTEED_PER_CATEGORY = 1

        selected = {}
        selected_set = set()
        categories = list(FACTOR_CATEGORIES.keys())

        pass_records = [r for r in records if r.get('risk_passed')]

        # 第1轮: 每类至少取1个(得分最高)
        for category in categories:
            cat_records = [r for r in pass_records if r.get('category') == category]
            cat_records.sort(key=lambda x: x.get('score', -1), reverse=True)
            n_take = min(GUARANTEED_PER_CATEGORY, len(cat_records))
            for r in cat_records[:n_take]:
                selected_set.add(r['factor_name'])
                selected.setdefault(category, []).append(r['factor_name'])

        # 第2轮: 全局择优补齐到 TOTAL_FACTORS
        if len(selected_set) < TOTAL_FACTORS:
            remaining = [r for r in pass_records if r['factor_name'] not in selected_set]
            remaining.sort(key=lambda x: x.get('score', -1), reverse=True)
            slot = TOTAL_FACTORS - len(selected_set)
            for r in remaining[:slot]:
                selected_set.add(r['factor_name'])
                selected.setdefault(r['category'], []).append(r['factor_name'])

        self.logger.info(f"共选择 {len(selected_set)} 个因子, 覆盖 {len(selected)} 个类别")
        for cat, factors in selected.items():
            self.logger.info(f"  {cat}: {len(factors)} 个因子 -> {factors[:3]}...")
        return selected

    def run_stage2(self, selected_factors: Dict[str, List[str]], start_date: str, end_date: str) -> List[Dict]:
        """阶段2：多因子权重遗传优化。"""
        self.logger.info("=" * 60)
        self.logger.info("阶段2：多因子权重遗传优化")
        self.logger.info("=" * 60)

        all_factors = []
        for factors in selected_factors.values():
            all_factors.extend(factors)
        all_factors = list(set(all_factors))
        n_factors = len(all_factors)

        self.logger.info(f"共 {n_factors} 个因子参与权重优化")

        required = self._get_required_factors()
        factor_data = self.db.get_factors(
            factor_names=list(set(required + all_factors)),
            start_date=start_date,
            end_date=end_date,
            with_close=True,
        )

        if factor_data is None or factor_data.empty:
            self.logger.error("无法加载因子数据")
            return []

        population_size, generations = 30, 50
        population = [np.random.randn(n_factors) * 0.3 for _ in range(population_size)]

        best_weights, best_scores = [], []

        for gen in range(generations):
            scores = []
            for weights in population:
                score = self._evaluate_weights(weights, all_factors, factor_data, start_date, end_date)
                scores.append(score)

            indices = np.argsort(scores)[::-1]
            population = [population[i] for i in indices]
            scores = [scores[i] for i in indices]

            if scores[0] > 0:
                best_weights.append(population[0].copy())
                best_scores.append(scores[0])

            if gen % 10 == 0:
                self.logger.info(f"第 {gen}/{generations} 代: 最佳得分={scores[0]:.4f}")

            new_pop = population[:3].copy()
            while len(new_pop) < population_size:
                if np.random.random() < 0.7 and len(population) > 2:
                    p1, p2 = np.random.choice(len(population) // 2, 2, replace=False)
                    child = (population[p1] + population[p2]) / 2
                else:
                    child = population[np.random.randint(len(population) // 2)].copy()

                if np.random.random() < 0.1:
                    child += np.random.randn(n_factors) * 0.2

                new_pop.append(child)

            population = new_pop

        sorted_idx = np.argsort(best_scores)[::-1][:5]
        results = []
        for i, idx in enumerate(sorted_idx):
            weights_dict = {all_factors[j]: float(best_weights[idx][j]) for j in range(n_factors)}
            non_zero = {k: v for k, v in weights_dict.items() if abs(v) > 0.01}

            results.append({
                'name': f'config_{i+1}',
                'score': float(best_scores[idx]),
                'weights': weights_dict,
                'non_zero_count': len(non_zero),
            })

            top_weights = sorted(non_zero.items(), key=lambda x: -abs(x[1]))[:5]
            self.logger.info(f"配置{i+1}: 得分={best_scores[idx]:.4f}, 非零因子={len(non_zero)}")
            self.logger.info(f"  主要权重: {dict(top_weights)}")

        return results

    def _evaluate_weights(self, weights, factor_names, factor_data, start_date, end_date):
        try:
            weights_dict = {factor_names[i]: float(weights[i]) for i in range(len(factor_names))}

            selector = MultiFactorSelector(weights=weights_dict, normalize_weights=False)

            if factor_data is None or factor_data.empty:
                return -1

            strategy = SignalStrategy(
                name="多因子-权重评估",
                selector=selector,
                position_sizer=TrendVolatilityTiming(),
                composer=LayeredComposer(top_n=30),
                top_n=30,
            )

            engine = BacktestEngine()
            report = engine.run(
                strategy=strategy,
                factor_data=factor_data,
                start_date=start_date,
                end_date=end_date,
            )

            calmar = report.annualized_return / abs(report.max_drawdown) if report.max_drawdown != 0 else 0

            risk_result = self.risk_constraints.check_backtest_result(
                annual_return=report.annualized_return,
                max_drawdown=abs(report.max_drawdown),
                volatility=getattr(report, "volatility", 0.5),
                calmar_ratio=calmar,
                win_rate=report.win_rate,
                turnover=getattr(report, "turnover", None),
            )

            if not risk_result.passed:
                return -1

            return (report.annualized_return * 0.3 + report.sharpe_ratio * 0.25 +
                    calmar * 0.3 + report.win_rate * 0.15)

        except Exception as e:
            return -1

    def run_full(self, start_date: str = "2019-01-01", end_date: str = None):
        """执行完整优化流程。"""
        if end_date is None:
            end_date = self.db.get_max_date('daily_bars')

        self.logger.info("=" * 60)
        self.logger.info("分类因子优化 - 完整流程")
        self.logger.info(f"回测区间: {start_date} ~ {end_date}")
        self.logger.info(f"输出目录: {self.today_dir}")
        self.logger.info("=" * 60)

        selected = self.run_stage1(start_date, end_date)

        best_weights = []
        if selected:
            stage2_start = max(start_date, '2021-01-01')
            best_weights = self.run_stage2(selected, stage2_start, end_date)

        config = {
            'timestamp': datetime.now().isoformat(),
            'selected_factors': selected,
            'best_weights': best_weights,
        }

        config_file = self.today_dir / "final_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        self.logger.info(f"\n优化完成! 配置已保存: {config_file}")

        return config


def main():
    import argparse
    parser = argparse.ArgumentParser(description="分类因子优化")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-dd", type=float, default=0.60)
    args = parser.parse_args()

    optimizer = DailyOptimizer(max_drawdown=args.max_dd)
    optimizer.run_full(args.start, args.end)


if __name__ == "__main__":
    main()