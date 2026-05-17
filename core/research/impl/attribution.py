"""绩效归因体系。

提供策略收益的归因分析功能。

归因维度
--------
    - 择时归因: 仓位变化对收益的贡献
    - 选股归因: 选股Alpha对收益的贡献
    - 行业归因: 行业配置对收益的贡献
    - 风格归因: 风格暴露对收益的贡献

用法
----
    from core.research.impl.attribution import AttributionAnalyzer

    analyzer = AttributionAnalyzer(db)
    report = analyzer.analyze(
        portfolio_returns=returns,
        benchmark_returns=benchmark,
        positions=positions,
    )
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class AttributionResult:
    """归因结果。"""
    total_return: float
    timing_contribution: float
    selection_contribution: float
    interaction: float
    industry_contribution: float
    style_contribution: float
    residual: float

    def to_dict(self) -> Dict:
        return {
            "total_return": self.total_return,
            "timing_contribution": self.timing_contribution,
            "selection_contribution": self.selection_contribution,
            "interaction": self.interaction,
            "industry_contribution": self.industry_contribution,
            "style_contribution": self.style_contribution,
            "residual": self.residual,
        }


@dataclass
class BrinsonResult:
    """Brinson 归因结果。"""
    total_excess_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float

    def to_dict(self) -> Dict:
        return {
            "total_excess_return": self.total_excess_return,
            "allocation_effect": self.allocation_effect,
            "selection_effect": self.selection_effect,
            "interaction_effect": self.interaction_effect,
        }


class AttributionAnalyzer:
    """绩效归因分析器。"""

    def __init__(self, db=None):
        self.db = db

    def analyze(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        positions: pd.DataFrame,
    ) -> AttributionResult:
        """
        综合归因分析。

        Args:
            portfolio_returns: 组合收益序列
            benchmark_returns: 基准收益序列
            positions: 持仓数据

        Returns:
            AttributionResult
        """
        excess_returns = portfolio_returns - benchmark_returns

        timing_contrib = self._timing Attribution(portfolio_returns, benchmark_returns)
        selection_contrib = self._selection Attribution(portfolio_returns, benchmark_returns)
        industry_contrib = self._industry Attribution(positions)
        style_contrib = self._style Attribution(positions)

        total_return = portfolio_returns.sum()

        residual = total_return - (
            timing_contrib + selection_contrib + industry_contrib + style_contrib
        )

        return AttributionResult(
            total_return=total_return,
            timing_contribution=timing_contrib,
            selection_contribution=selection_contrib,
            interaction=0,
            industry_contribution=industry_contrib,
            style_contribution=style_contrib,
            residual=residual,
        )

    def _timing Attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        择时归因。

        Brinson 模型：超额收益 = 配置效应 + 选择效应 + 交互效应
        择时归因关注配置效应。
        """
        avg_portfolio = portfolio_returns.mean()
        avg_benchmark = benchmark_returns.mean()

        timing_return = avg_portfolio - avg_benchmark

        return timing_return

    def _selection Attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        选股归因。

        选股归因关注选择效应。
        """
        portfolio_vol = portfolio_returns.std()
        benchmark_vol = benchmark_returns.std()

        selection_return = (portfolio_vol - benchmark_vol) * 0.5

        return selection_return

    def _industry Attribution(self, positions: pd.DataFrame) -> float:
        """行业归因。"""
        if "industry" not in positions.columns:
            return 0.0

        industry_returns = positions.groupby("industry")["return"].mean()
        industry_weights = positions.groupby("industry")["weight"].mean()

        contribution = (industry_returns * industry_weights).sum()

        return contribution

    def _style Attribution(self, positions: pd.DataFrame) -> float:
        """风格归因。"""
        if "style" not in positions.columns:
            return 0.0

        style_returns = positions.groupby("style")["return"].mean()
        style_weights = positions.groupby("style")["weight"].mean()

        contribution = (style_returns * style_weights).sum()

        return contribution

    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        group_column: str = "industry",
    ) -> BrinsonResult:
        """
        Brinson 归因分析。

        Args:
            portfolio_weights: 组合权重
            benchmark_weights: 基准权重
            portfolio_returns: 组合收益
            benchmark_returns: 基准收益
            group_column: 分组列名

        Returns:
            BrinsonResult
        """
        pf_w = portfolio_weights.set_index(group_column)["weight"]
        bm_w = benchmark_weights.set_index(group_column)["weight"]
        pf_r = portfolio_returns
        bm_r = benchmark_returns

        all_groups = pf_w.index.union(bm_w.index)
        pf_w = pf_w.reindex(all_groups, fill_value=0)
        bm_w = bm_w.reindex(all_groups, fill_value=0)

        allocation_effect = ((pf_w - bm_w) * bm_r.values).sum()
        selection_effect = (bm_w * (pf_r.values - bm_r.values)).sum()
        interaction_effect = ((pf_w - bm_w) * (pf_r.values - bm_r.values)).sum()

        total_excess = portfolio_returns.mean() - benchmark_returns.mean()

        return BrinsonResult(
            total_excess_return=total_excess,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
        )

    def factor Attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series],
        exposures: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        因子归因。

        使用 Barra 风格的因子归因。

        Args:
            portfolio_returns: 组合收益
            factor_returns: 因子收益字典
            exposures: 因子暴露

        Returns:
            各因子贡献的字典
        """
        contributions = {}

        for factor_name, factor_ret in factor_returns.items():
            if factor_name not in exposures.columns:
                continue

            exposure = exposures[factor_name].mean()

            aligned_ret = factor_ret.reindex(portfolio_returns.index).fillna(0)

            contribution = (exposure * aligned_ret).sum()
            contributions[factor_name] = contribution

        total_contribution = sum(contributions.values())
        unexplained = portfolio_returns.sum() - total_contribution
        contributions["residual"] = unexplained

        return contributions


class RiskAnalyzer:
    """风险分析器。"""

    def __init__(self):
        pass

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """
        计算风险指标。

        Returns:
            风险指标字典
        """
        return {
            "volatility": returns.std() * np.sqrt(252),
            "var_95": returns.quantile(0.05) * np.sqrt(252),
            "cvar_95": returns[returns <= returns.quantile(0.05)].mean() * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(returns),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤。"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def factor_risk_contribution(
        self,
        exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        计算因子风险贡献。

        Args:
            exposures: 因子暴露
            factor_covariance: 因子协方差矩阵

        Returns:
            各因子风险贡献
        """
        portfolio_variance = exposures.dot(factor_covariance).dot(exposures.T)
        risk_contributions = exposures.multiply(
            exposures.dot(factor_covariance), axis=0
        )
        marginal_contributions = risk_contributions.div(portfolio_variance.sum())
        return marginal_contributions.sum(axis=1)
