"""遗传算法权重优化模块（第二阶段）。

用筛选出的因子，通过遗传算法优化出5个最佳权重配比。
"""
from __future__ import annotations
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from core.database import Database
from core.backtest import BacktestEngine, BacktestReport
from core.screening import MultiFactorSelector
from core.timings import TrendVolatilityTiming
from core.positioners import EqualWeightBuilder
from core.strategy import QuantStrategy
from core.optimization.risk_constraints import RiskConstraints

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """权重配置。"""
    name: str
    weights: Dict[str, float]
    report: BacktestReport
    risk_check_passed: bool
    score: float

    @property
    def key_metrics(self) -> Dict:
        calmar = self.report.annual_return / self.report.max_drawdown if self.report.max_drawdown > 0 else 0
        return {
            "name": self.name,
            "total_return": self.report.total_return,
            "annual_return": self.report.annual_return,
            "max_drawdown": self.report.max_drawdown,
            "sharpe_ratio": self.report.sharpe_ratio,
            "calmar_ratio": calmar,
            "win_rate": self.report.win_rate,
            "risk_passed": self.risk_check_passed,
            "score": self.score,
        }


class GeneticWeightOptimizer:
    """遗传算法权重优化器。"""

    def __init__(
        self,
        db: Database,
        risk_constraints: RiskConstraints,
        factor_names: List[str],
        top_n: int = 30,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_size: int = 3,
    ):
        self.db = db
        self.risk_constraints = risk_constraints
        self.factor_names = factor_names
        self.top_n = top_n
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size

        self.results: List[WeightConfig] = []
        self.best_history: List[float] = []

    def run(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2023-12-31",
        target_config_count: int = 5,
    ) -> List[WeightConfig]:
        """运行遗传算法优化。"""
        logger.info(
            f"开始权重优化: {len(self.factor_names)} 个因子, "
            f"{self.generations} 代, 种群大小 {self.population_size}"
        )

        population = self._initialize_population()

        for gen in range(self.generations):
            logger.info(f"第 {gen+1}/{self.generations} 代...")

            fitness_scores = self._evaluate_population(
                population, start_date, end_date
            )

            avg_fitness = np.mean([f for f in fitness_scores if f > 0])
            best_fitness = max(fitness_scores)
            self.best_history.append(best_fitness)

            logger.info(
                f"  平均适应度: {avg_fitness:.4f}, 最佳: {best_fitness:.4f}"
            )

            population = self._next_generation(population, fitness_scores)

        logger.info("进化完成！开始回测最终配置...")

        final_population = population
        final_scores = self._evaluate_population(
            final_population, start_date, end_date
        )

        all_configs = []
        for i, (weights, score) in enumerate(zip(final_population, final_scores)):
            if score > 0:
                config = self._weights_to_config(
                    weights, f"配置-{i+1}", start_date, end_date
                )
                if config is not None:
                    all_configs.append(config)

        all_configs.sort(key=lambda c: c.score, reverse=True)
        self.results = all_configs

        return all_configs[:target_config_count]

    def _initialize_population(self) -> List[np.ndarray]:
        """初始化种群。"""
        population = []
        n_factors = len(self.factor_names)

        for _ in range(self.population_size):
            weights = np.random.normal(0, 0.5, n_factors)

            random_mask = np.random.random(n_factors) < 0.3
            weights[random_mask] = 0

            if np.all(weights == 0):
                weights[np.random.randint(n_factors)] = 1.0

            population.append(weights)

        return population

    def _evaluate_population(
        self,
        population: List[np.ndarray],
        start_date: str,
        end_date: str,
    ) -> List[float]:
        """评估种群适应度。"""
        scores = []

        for weights in population:
            score = self._evaluate_weights(weights, start_date, end_date)
            scores.append(score)

        return scores

    def _evaluate_weights(
        self,
        weights: np.ndarray,
        start_date: str,
        end_date: str,
    ) -> float:
        """评估一组权重的适应度。"""
        try:
            weight_dict = dict(zip(self.factor_names, weights))

            selector = MultiFactorSelector(weight_dict)
            timing = TrendVolatilityTiming()
            portfolio = EqualWeightBuilder()

            strategy = QuantStrategy(
                name="多因子-权重优化",
                selector=selector,
                timing=timing,
                portfolio=portfolio,
                top_n=self.top_n,
            )

            required_factors = ['momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal', 'volatility_20', 'volume_ratio', 'boll_position']
            all_factors = list(set(required_factors + self.factor_names))

            factor_data = self.db.get_factors(
                factor_names=all_factors,
                start_date=start_date,
                end_date=end_date,
                with_close=True,
            )

            if factor_data is None or factor_data.empty:
                return -1

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
            )

            if not risk_result.passed:
                return -1

            return (
                report.annualized_return * 0.3 +
                report.sharpe_ratio * 0.25 +
                calmar * 0.3 +
                report.win_rate * 0.15
            )

        except Exception as e:
            logger.debug(f"权重评估失败: {e}")
            return -1

    def _weights_to_config(
        self,
        weights: np.ndarray,
        name: str,
        start_date: str,
        end_date: str,
    ) -> Optional[WeightConfig]:
        """将权重转为配置对象（完整回测）。"""
        try:
            weight_dict = dict(zip(self.factor_names, weights))

            selector = MultiFactorSelector(weight_dict)
            timing = TrendVolatilityTiming()
            portfolio = EqualWeightBuilder()

            strategy = QuantStrategy(
                name=f"多因子-{name}",
                selector=selector,
                timing=timing,
                portfolio=portfolio,
                top_n=self.top_n,
            )

            required_factors = ['momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal', 'volatility_20', 'volume_ratio', 'boll_position']
            all_factors = list(set(required_factors + self.factor_names))

            factor_data = self.db.get_factors(
                factor_names=all_factors,
                start_date=start_date,
                end_date=end_date,
                with_close=True,
            )

            if factor_data is None or factor_data.empty:
                return None

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
            )

            score = (
                report.annualized_return * 0.3 +
                report.sharpe_ratio * 0.25 +
                calmar * 0.3 +
                report.win_rate * 0.15
            ) if risk_result.passed else -1

            return WeightConfig(
                name=name,
                weights=weight_dict,
                report=report,
                risk_check_passed=risk_result.passed,
                score=score,
            )

        except Exception as e:
            logger.error(f"权重转配置失败: {e}")
            return None

    def _next_generation(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
    ) -> List[np.ndarray]:
        """生成下一代。"""
        n = len(population)

        combined = list(zip(fitness_scores, population))
        combined.sort(key=lambda x: x[0], reverse=True)

        elites = [x[1] for x in combined[:self.elitism_size]]

        new_population = elites.copy()

        while len(new_population) < n:
            parent1 = self._selection(population, fitness_scores)
            parent2 = self._selection(population, fitness_scores)

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)

            new_population.append(child1)
            if len(new_population) < n:
                new_population.append(child2)

        return new_population[:n]

    def _selection(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
    ) -> np.ndarray:
        """轮盘赌选择。"""
        valid = [
            (i, f) for i, f in enumerate(fitness_scores)
            if f > 0
        ]
        if not valid:
            return population[random.randint(0, len(population)-1)]

        valid_indices = [i for i, f in valid]
        weights = np.array([f for i, f in valid])
        weights = weights - weights.min() + 1e-6
        probs = weights / weights.sum()

        selected_idx = random.choices(valid_indices, weights=probs, k=1)[0]
        return population[selected_idx]

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单点交叉。"""
        point = random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, weights: np.ndarray) -> np.ndarray:
        """变异。"""
        new_weights = weights.copy()

        for i in range(len(new_weights)):
            if random.random() < 0.1:
                new_weights[i] = np.random.normal(0, 0.5)

            if random.random() < 0.05:
                new_weights[i] = 0

        if np.all(new_weights == 0):
            new_weights[random.randint(0, len(new_weights)-1)] = 1.0

        return new_weights

    def get_results_df(self) -> pd.DataFrame:
        """结果转 DataFrame。"""
        if not self.results:
            return pd.DataFrame()
        rows = [r.key_metrics for r in self.results]
        df = pd.DataFrame(rows)
        return df.sort_values("score", ascending=False).reset_index(drop=True)
