"""因子筛选模块（第一阶段）。

从所有因子中筛选出30个最佳因子，用单因子选股器 + 趋势波动率二重复合择时回测。
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from ...database import Database
from ...screening import FactorRankSelector
from ...timings import TrendVolatilityTiming
from ...execution.impl.backtest import BacktestEngine, BacktestReport
from ...strategies.base.strategy import SignalStrategy
from ...signals.base.composer import LayeredComposer
from ..base.risk_constraints import RiskConstraints

logger = logging.getLogger(__name__)


@dataclass
class FactorBacktestResult:
    """单个因子的回测结果。"""
    factor_name: str
    report: BacktestReport
    risk_check_passed: bool
    score: float

    @property
    def key_metrics(self) -> Dict:
        calmar_ratio = self.report.annualized_return / abs(self.report.max_drawdown) if self.report.max_drawdown != 0 else 0
        return {
            "factor_name": self.factor_name,
            "total_return": self.report.total_return,
            "annual_return": self.report.annualized_return,
            "max_drawdown": abs(self.report.max_drawdown),
            "sharpe_ratio": self.report.sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "win_rate": self.report.win_rate,
            "turnover": getattr(self.report, "turnover", np.nan),
            "risk_passed": self.risk_check_passed,
            "score": self.score,
        }


class FactorSelector:
    """因子筛选器（第一阶段）。"""

    def __init__(
        self,
        db: Database,
        risk_constraints: RiskConstraints,
        top_n: int = 30,
        target_factor_count: int = 30,
    ):
        self.db = db
        self.risk_constraints = risk_constraints
        self.top_n = top_n
        self.target_factor_count = target_factor_count
        self.results: List[FactorBacktestResult] = []

    def run(
        self,
        factor_names: Optional[List[str]] = None,
        start_date: str = "2019-01-01",
        end_date: str = "2023-12-31",
        parallel: bool = True,
    ) -> List[FactorBacktestResult]:
        """运行因子筛选流程。

        Args:
            factor_names: 要筛选的因子列表（None表示全部）
            start_date: 回测起始日期
            end_date: 回测结束日期
            parallel: 是否并行运行

        Returns:
            List[FactorBacktestResult]: 所有因子的回测结果（按得分排序）
        """
        if factor_names is None:
            factor_names = self.db.list_factor_columns()

        logger.info(f"开始筛选 {len(factor_names)} 个因子...")

        if parallel and len(factor_names) > 10:
            self.results = self._run_parallel(factor_names, start_date, end_date)
        else:
            self.results = self._run_sequential(factor_names, start_date, end_date)

        self.results.sort(key=lambda r: r.score, reverse=True)
        return self.results

    def _run_sequential(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
    ) -> List[FactorBacktestResult]:
        """串行回测。"""
        results = []
        for i, factor in enumerate(factor_names, 1):
            logger.info(f"[{i}/{len(factor_names)}] 回测因子: {factor}")
            result = self._backtest_single_factor(factor, start_date, end_date)
            if result is not None:
                results.append(result)
        return results

    def _run_parallel(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
    ) -> List[FactorBacktestResult]:
        """并行回测。"""
        results = []
        db_path = self.db.db_path

        def worker(factor: str):
            worker_db = Database(db_path)
            try:
                return _backtest_single_factor_static(
                    worker_db, factor, start_date, end_date,
                    self.top_n, self.risk_constraints
                )
            finally:
                worker_db.conn.close()

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(worker, f): f for f in factor_names}
            for i, future in enumerate(as_completed(futures), 1):
                factor = futures[future]
                logger.info(f"[{i}/{len(factor_names)}] 完成因子: {factor}")
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"因子 {factor} 回测失败: {e}")

        return results

    def _backtest_single_factor(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
    ) -> Optional[FactorBacktestResult]:
        """回测单个因子。"""
        try:
            selector = FactorRankSelector(factor_name=factor_name, ascending=False)
            timing = TrendVolatilityTiming()

            strategy = SignalStrategy(
                name=f"单因子-{factor_name}",
                selector=selector,
                position_sizer=timing,
                composer=LayeredComposer(top_n=self.top_n),
                top_n=self.top_n,
            )

            required_factors = ['momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal', 'volatility_20', 'volume_ratio', 'boll_position']
            all_factors = list(set(required_factors + [factor_name]))

            factor_data = self.db.get_factors(
                factor_names=all_factors,
                start_date=start_date,
                end_date=end_date,
                with_close=True,
            )

            if factor_data is None or factor_data.empty:
                logger.warning(f"因子 {factor_name} 没有数据")
                return None

            engine = BacktestEngine()
            report = engine.run(
                strategy=strategy,
                factor_data=factor_data,
                start_date=start_date,
                end_date=end_date,
            )

            calmar_ratio = report.annualized_return / abs(report.max_drawdown) if report.max_drawdown != 0 else 0

            risk_result = self.risk_constraints.check_backtest_result(
                annual_return=report.annualized_return,
                max_drawdown=abs(report.max_drawdown),
                volatility=getattr(report, "volatility", 0.5),
                calmar_ratio=calmar_ratio,
                win_rate=report.win_rate,
                turnover=getattr(report, "turnover", None),
            )

            score = self._calculate_score(report, calmar_ratio, risk_result.passed)

            return FactorBacktestResult(
                factor_name=factor_name,
                report=report,
                risk_check_passed=risk_result.passed,
                score=score,
            )

        except Exception as e:
            logger.error(f"因子 {factor_name} 回测异常: {e}")
            return None

    def _calculate_score(
        self,
        report: BacktestReport,
        calmar_ratio: float,
        risk_passed: bool,
    ) -> float:
        """计算因子综合得分（可通过子类覆盖权重）。"""
        if not risk_passed:
            return -1

        # 权重体系: 年化收益30% + Sharpe25% + Calmar30% + 胜率15%
        return (
            report.annualized_return * 0.3 +
            report.sharpe_ratio * 0.25 +
            calmar_ratio * 0.3 +
            report.win_rate * 0.15
        )

    def get_top_factors(self) -> List[FactorBacktestResult]:
        """获取最佳的N个因子（先风控通过，再按得分排序）。"""
        passed = [r for r in self.results if r.risk_check_passed]
        passed.sort(key=lambda r: r.score, reverse=True)
        return passed[:self.target_factor_count]

    def get_results_df(self) -> pd.DataFrame:
        """将结果转为 DataFrame。"""
        if not self.results:
            return pd.DataFrame()
        rows = [r.key_metrics for r in self.results]
        df = pd.DataFrame(rows)
        return df.sort_values("score", ascending=False).reset_index(drop=True)


def _backtest_single_factor_static(
    db: Database,
    factor_name: str,
    start_date: str,
    end_date: str,
    top_n: int,
    risk_constraints: RiskConstraints,
) -> Optional[FactorBacktestResult]:
    """静态函数，用于并行。"""
    try:
        selector = FactorRankSelector(factor_name=factor_name, ascending=False)
        timing = TrendVolatilityTiming()

        strategy = SignalStrategy(
            name=f"单因子-{factor_name}",
            selector=selector,
            position_sizer=timing,
            composer=LayeredComposer(top_n=top_n),
            top_n=top_n,
        )

        required_factors = ['momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal', 'volatility_20', 'volume_ratio', 'boll_position']
        all_factors = list(set(required_factors + [factor_name]))

        factor_data = db.get_factors(
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

        calmar_ratio = report.annualized_return / abs(report.max_drawdown) if report.max_drawdown != 0 else 0

        risk_result = risk_constraints.check_backtest_result(
            annual_return=report.annualized_return,
            max_drawdown=abs(report.max_drawdown),
            volatility=getattr(report, "volatility", 0.5),
            calmar_ratio=calmar_ratio,
            win_rate=report.win_rate,
            turnover=getattr(report, "turnover", None),
        )

        score = (
            report.annualized_return * 0.3 +
            report.sharpe_ratio * 0.25 +
            calmar_ratio * 0.3 +
            report.win_rate * 0.15
        ) if risk_result.passed else -1

        return FactorBacktestResult(
            factor_name=factor_name,
            report=report,
            risk_check_passed=risk_result.passed,
            score=score,
        )

    except Exception as e:
        logger.error(f"因子 {factor_name} 回测异常: {e}")
        return None
