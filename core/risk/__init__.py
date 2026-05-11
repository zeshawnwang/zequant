"""风控管理器模块。

提供完整的风险管理功能。

风控功能
--------
    - 仓位约束: 单票上限、总仓位上限、最小仓位
    - 止损止盈: 固定止损、移动止损、时间止损
    - 风险预警: VaR、CVaR、最大回撤
    - 交易约束: 换手率限制、流动性约束

用法
----
    from core.risk import RiskManager, StopLoss, TakeProfit

    risk_manager = RiskManager(
        constraints=[
            MaxPositionConstraint(max_total=0.9),
            MinPositionConstraint(min_single=0.001),
        ],
        stop_loss=StopLoss(method="fixed", threshold=0.1),
        take_profit=TakeProfit(method="trailing", threshold=0.25),
    )

    managed_weights = risk_manager.apply(weights, cash, positions)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np


class IConstraint(ABC):
    """风控约束接口。"""

    @abstractmethod
    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        """应用约束。"""
        pass


class MaxPositionConstraint(IConstraint):
    """总仓位上限约束。"""

    def __init__(self, max_total: float = 0.9):
        self.max_total = max_total

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        total = sum(weights.values())
        if total > self.max_total:
            scale = self.max_total / total
            return {k: v * scale for k, v in weights.items()}
        return weights


class MinPositionConstraint(IConstraint):
    """最小仓位约束。"""

    def __init__(self, min_single: float = 0.001, min_total: float = 0.0):
        self.min_single = min_single
        self.min_total = min_total

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        filtered = {k: v for k, v in weights.items() if v >= self.min_single}
        total = sum(filtered.values())
        if total < self.min_total and filtered:
            scale = self.min_total / total
            filtered = {k: v * scale for k, v in filtered.items()}
        return filtered


class SingleWeightConstraint(IConstraint):
    """单票权重上限约束。"""

    def __init__(self, max_single: float = 0.05):
        self.max_single = max_single

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        return {k: min(v, self.max_single) for k, v in weights.items()}


class TurnoverConstraint(IConstraint):
    """换手率约束。"""

    def __init__(self, max_turnover: float = 0.3):
        self.max_turnover = max_turnover

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        if not positions:
            return weights

        all_symbols = set(weights.keys()) | set(positions.keys())
        total_weight = sum(weights.values())

        turnover = sum(
            abs(weights.get(s, 0) - positions.get(s, 0)) for s in all_symbols
        ) / 2

        if turnover > self.max_turnover and total_weight > 0:
            scale = self.max_turnover / turnover
            return {k: v * scale for k, v in weights.items()}

        return weights


@dataclass
class StopLoss:
    """止损规则。"""
    method: str = "fixed"
    threshold: float = 0.10
    lookback: int = 20

    def check(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        prices: Optional[pd.Series] = None,
    ) -> bool:
        """检查是否触发止损。"""
        if self.method == "fixed":
            return (entry_price - current_price) / entry_price > self.threshold

        elif self.method == "trailing":
            if prices is None:
                return False
            peak_price = prices.max()
            return (peak_price - current_price) / peak_price > self.threshold

        return False


@dataclass
class TakeProfit:
    """止盈规则。"""
    method: str = "fixed"
    threshold: float = 0.25
    lookback: int = 20

    def check(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        prices: Optional[pd.Series] = None,
    ) -> bool:
        """检查是否触发止盈。"""
        if self.method == "fixed":
            return (current_price - entry_price) / entry_price > self.threshold

        elif self.method == "trailing":
            if prices is None:
                return False
            trough_price = prices.min()
            return (current_price - trough_price) / trough_price > self.threshold

        return False


@dataclass
class RiskMetrics:
    """风险指标。"""
    total_exposure: float
    max_single_position: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float


class RiskManager:
    """
    风控管理器。

    组合多个约束和规则，统一管理风险。
    """

    def __init__(
        self,
        constraints: Optional[List[IConstraint]] = None,
        stop_loss: Optional[StopLoss] = None,
        take_profit: Optional[TakeProfit] = None,
        max_total_exposure: float = 0.95,
        max_single_position: float = 0.10,
    ):
        """
        Args:
            constraints: 约束列表
            stop_loss: 止损规则
            take_profit: 止盈规则
            max_total_exposure: 最大总仓位
            max_single_position: 最大单票仓位
        """
        self.constraints = constraints or []
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_total_exposure = max_total_exposure
        self.max_single_position = max_single_position

        if max_single_position:
            self.constraints.append(SingleWeightConstraint(max_single_position))

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        """应用所有风控约束。"""
        result = weights.copy()

        for constraint in self.constraints:
            result = constraint.apply(result, cash, positions)

        return result

    def check_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        prices: Optional[pd.Series] = None,
    ) -> bool:
        """检查止损。"""
        if self.stop_loss is None:
            return False
        return self.stop_loss.check(symbol, entry_price, current_price, prices)

    def check_take_profit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        prices: Optional[pd.Series] = None,
    ) -> bool:
        """检查止盈。"""
        if self.take_profit is None:
            return False
        return self.take_profit.check(symbol, entry_price, current_price, prices)

    def calculate_metrics(
        self,
        returns: pd.Series,
        weights: Dict[str, float],
    ) -> RiskMetrics:
        """计算风险指标。"""
        total_exposure = sum(weights.values())

        max_single = max(weights.values()) if weights else 0

        var_95 = returns.quantile(0.05) if not returns.empty else 0
        cvar_95 = returns[returns <= var_95].mean() if not returns.empty else 0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0

        volatility = returns.std() * np.sqrt(252) if not returns.empty else 0

        mean_return = returns.mean() * 252 if not returns.empty else 0
        sharpe = mean_return / volatility if volatility > 0 else 0

        return RiskMetrics(
            total_exposure=total_exposure,
            max_single_position=max_single,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe,
        )

    def check_risk_limits(
        self,
        weights: Dict[str, float],
        returns: Optional[pd.Series] = None,
    ) -> List[str]:
        """检查风险限制，返回违规列表。"""
        warnings = []

        total_exposure = sum(weights.values())
        if total_exposure > self.max_total_exposure:
            warnings.append(
                f"总仓位 {total_exposure:.2%} 超过上限 {self.max_total_exposure:.2%}"
            )

        max_single = max(weights.values()) if weights else 0
        if max_single > self.max_single_position:
            warnings.append(
                f"单票仓位 {max_single:.2%} 超过上限 {self.max_single_position:.2%}"
            )

        if returns is not None and not returns.empty:
            daily_var = returns.quantile(0.05)
            if daily_var < -0.05:
                warnings.append(f"日 VaR {daily_var:.2%} 超过 -5%")

            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            current_dd = drawdown.iloc[-1] if not drawdown.empty else 0
            if current_dd < -0.20:
                warnings.append(f"当前回撤 {current_dd:.2%} 超过 -20%")

        return warnings


class RiskBudget:
    """风险预算管理。"""

    def __init__(
        self,
        total_budget: float = 0.20,
        per_position_limit: float = 0.05,
    ):
        self.total_budget = total_budget
        self.per_position_limit = per_position_limit
        self.used_budget = 0

    def allocate(
        self,
        symbol: str,
        volatility: float,
        target_weight: float,
    ) -> float:
        """
        分配风险预算。

        Args:
            symbol: 股票代码
            volatility: 波动率
            target_weight: 目标权重

        Returns:
            调整后的权重
        """
        risk_contribution = target_weight * volatility

        if self.used_budget + risk_contribution > self.total_budget:
            available = self.total_budget - self.used_budget
            adjusted_weight = available / volatility if volatility > 0 else 0
            return min(adjusted_weight, self.per_position_limit)

        self.used_budget += risk_contribution
        return min(target_weight, self.per_position_limit)

    def reset(self):
        """重置风险预算。"""
        self.used_budget = 0
