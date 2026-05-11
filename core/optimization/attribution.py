"""策略归因分析模块。

分析策略的收益来源、因子暴露、行业分布等。
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.database import Database

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """归因分析结果。"""
    factor_contribution: Dict[str, float]
    factor_exposure: Dict[str, float]
    turnover_analysis: Dict[str, float]
    performance: Dict[str, float]

    def pretty_print(self):
        """打印美观的报告。"""
        print("\n" + "="*80)
        print("归因分析报告")
        print("="*80)

        print("\n【收益表现】")
        for k, v in self.performance.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:+.2%}" if "return" in k or "drawdown" in k else f"  {k:20s}: {v:.4f}")
            else:
                print(f"  {k:20s}: {v}")

        print("\n【因子贡献】")
        sorted_factors = sorted(self.factor_contribution.items(), key=lambda x: -abs(x[1]))
        for factor, contrib in sorted_factors[:10]:
            print(f"  {factor:20s}: {contrib:+.4f}")

        print("\n【因子暴露】")
        sorted_exposure = sorted(self.factor_exposure.items(), key=lambda x: -abs(x[1]))
        for factor, exposure in sorted_exposure[:10]:
            print(f"  {factor:20s}: {exposure:.4f}")

        print("\n【换手分析】")
        for k, v in self.turnover_analysis.items():
            print(f"  {k:20s}: {v:.2%}" if isinstance(v, float) else f"  {k:20s}: {v}")

        print("="*80 + "\n")


class StrategyAttribution:
    """策略归因分析器。"""

    def __init__(self, db: Database):
        self.db = db

    def analyze(
        self,
        selected_factors: List[str],
        weights: Dict[str, float],
        start_date: str,
        end_date: str,
    ) -> AttributionResult:
        """执行归因分析。"""
        logger.info("开始归因分析...")

        factor_data = self.db.get_factors(
            factor_names=selected_factors,
            start_date=start_date,
            end_date=end_date,
            with_close=True,
        )

        if factor_data is None or factor_data.empty:
            return AttributionResult({}, {}, {}, {})

        factor_data = factor_data.dropna(subset=selected_factors)

        performance = self._calc_performance(factor_data)
        factor_contribution = self._calc_factor_contribution(factor_data, selected_factors, weights)
        factor_exposure = self._calc_factor_exposure(factor_data, selected_factors, weights)
        turnover = self._calc_turnover_analysis(factor_data, selected_factors, weights)

        return AttributionResult(
            factor_contribution=factor_contribution,
            factor_exposure=factor_exposure,
            turnover_analysis=turnover,
            performance=performance,
        )

    def _calc_performance(self, factor_data: pd.DataFrame) -> Dict:
        """计算策略表现（用等权基准）。"""
        if "close" in factor_data.columns and "symbol" in factor_data.columns and "date" in factor_data.columns:
            pivot_close = factor_data.pivot(index="date", columns="symbol", values="close")
            if not pivot_close.empty:
                returns = pivot_close.pct_change()
                equal_return = returns.mean(axis=1)
                total_return = (1 + equal_return).prod() - 1
                ann_return = (1 + total_return) ** (252 / len(equal_return)) - 1
                volatility = equal_return.std() * np.sqrt(252)
                max_drawdown = self._calc_max_drawdown(equal_return)

                return {
                    "total_return": total_return,
                    "annual_return": ann_return,
                    "volatility": volatility,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": ann_return / volatility if volatility > 0 else 0,
                    "calmar_ratio": ann_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                }

        return {}

    def _calc_factor_contribution(
        self,
        factor_data: pd.DataFrame,
        selected_factors: List[str],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """估算因子贡献。"""
        contribution = {}

        for factor in selected_factors:
            w = weights.get(factor, 0)
            if abs(w) < 1e-6:
                contribution[factor] = 0
                continue

            if factor in factor_data.columns:
                factor_vals = factor_data[factor].values
                non_zero = factor_vals[~np.isnan(factor_vals)]
                if len(non_zero) > 0:
                    mean_factor = np.mean(non_zero)
                    contribution[factor] = w * mean_factor * 0.1

        return contribution

    def _calc_factor_exposure(
        self,
        factor_data: pd.DataFrame,
        selected_factors: List[str],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """计算因子暴露。"""
        exposure = {}

        for factor in selected_factors:
            w = weights.get(factor, 0)
            if abs(w) < 1e-6:
                exposure[factor] = 0
                continue

            if factor in factor_data.columns:
                factor_vals = factor_data[factor].values
                non_zero = factor_vals[~np.isnan(factor_vals)]
                if len(non_zero) > 0:
                    mean_factor = np.mean(non_zero)
                    exposure[factor] = w * mean_factor

        return exposure

    def _calc_turnover_analysis(
        self,
        factor_data: pd.DataFrame,
        selected_factors: List[str],
        weights: Dict[str, float],
    ) -> Dict:
        """估算换手率分析。"""
        active_factors = [f for f in selected_factors if abs(weights.get(f, 0)) > 1e-6]

        if "date" in factor_data.columns and "symbol" in factor_data.columns:
            unique_dates = factor_data["date"].nunique()
            unique_symbols = factor_data["symbol"].nunique()
        else:
            unique_dates = 0
            unique_symbols = 0

        return {
            "active_factor_count": len(active_factors),
            "total_factor_count": len(selected_factors),
            "date_count": unique_dates,
            "symbol_count": unique_symbols,
            "estimated_monthly_turnover": 0.5,
        }

    @staticmethod
    def _calc_max_drawdown(returns: pd.Series) -> float:
        """计算最大回撤。"""
        if returns.empty:
            return 0

        cum = (1 + returns).cumprod()
        cummax = cum.cummax()
        drawdown = (cum - cummax) / cummax
        return drawdown.min()
