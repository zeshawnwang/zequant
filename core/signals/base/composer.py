"""信号组合器基类。

包含：
  - IComposer: 组合器接口
  - LayeredComposer: 分层组合器
  - DirectComposer: 直接组合器
  - WeightedComposer: 加权组合器
  - VoteComposer: 投票组合器
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass


class IConstraint(ABC):
    """约束接口。"""
    @abstractmethod
    def apply(self, weights: Dict[str, float], cash: float) -> Dict[str, float]:
        """应用约束。"""
        pass


@dataclass
class MaxSingleWeightConstraint(IConstraint):
    """单票最大权重约束。"""
    max_weight: float = 0.1

    def apply(self, weights: Dict[str, float], cash: float) -> Dict[str, float]:
        result = {}
        for symbol, weight in weights.items():
            result[symbol] = min(weight, self.max_weight)
        total = sum(result.values())
        if total > 0 and abs(total - 1.0) > 1e-10:
            factor = 1.0 / total
            result = {k: v * factor for k, v in result.items()}
        return result


@dataclass
class MaxTotalPositionConstraint(IConstraint):
    """总仓位最大权重约束。"""
    max_position: float = 0.9

    def apply(self, weights: Dict[str, float], cash: float) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= self.max_position:
            return weights

        factor = self.max_position / total
        return {k: v * factor for k, v in weights.items()}


@dataclass
class ReserveCashConstraint(IConstraint):
    """预留现金约束。"""
    reserve_ratio: float = 0.1

    def apply(self, weights: Dict[str, float], cash: float) -> Dict[str, float]:
        factor = 1.0 - self.reserve_ratio
        return {k: v * factor for k, v in weights.items()}


class IComposer(ABC):
    """信号组合器接口。"""

    @abstractmethod
    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """组合信号生成权重。"""
        pass

    def apply_constraints(
        self,
        weights: Dict[str, float],
        cash: float,
    ) -> Dict[str, float]:
        """应用所有约束。"""
        if not hasattr(self, "constraints"):
            return weights

        for constraint in self.constraints:
            weights = constraint.apply(weights, cash)
        return weights


class LayeredComposer(IComposer):
    """分层组合器：先选股，再分配权重。"""

    def __init__(
        self,
        top_n: int = 30,
        constraints: Optional[List[IConstraint]] = None,
    ):
        self.top_n = top_n
        self.constraints = constraints or []

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        sorted_scores = sorted(
            selector_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_symbols = [s for s, _ in sorted_scores[: self.top_n]]

        if not top_symbols:
            return {}

        weight_per_stock = position_signal / len(top_symbols)
        weights = {symbol: weight_per_stock for symbol in top_symbols}

        return self.apply_constraints(weights, cash)


class DirectComposer(IComposer):
    """直接组合器：直接使用分数作为权重。"""

    def __init__(
        self,
        constraints: Optional[List[IConstraint]] = None,
    ):
        self.constraints = constraints or []

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        if not selector_scores:
            return {}

        total_score = sum(abs(v) for v in selector_scores.values())
        if total_score == 0:
            return {}

        weights = {}
        for symbol, score in selector_scores.items():
            raw_weight = score / total_score
            weights[symbol] = raw_weight * position_signal

        return self.apply_constraints(weights, cash)


class WeightedComposer(IComposer):
    """加权组合器：使用自定义权重。"""

    def __init__(
        self,
        weights: Dict[str, float],
        constraints: Optional[List[IConstraint]] = None,
    ):
        self.weights = weights
        self.constraints = constraints or []

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        result = {}
        for symbol, score in selector_scores.items():
            factor_weight = self.weights.get(symbol, 1.0)
            result[symbol] = score * factor_weight * position_signal

        return self.apply_constraints(result, cash)


class VoteComposer(IComposer):
    """投票组合器：多数投票决定。"""

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.thresholds = thresholds or {}

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        if not selector_scores:
            return {}

        buy_signals = {
            symbol: score for symbol, score in selector_scores.items()
            if score >= self.thresholds.get(symbol, 0.5)
        }

        n = len(buy_signals)
        if n == 0:
            return {}

        weight_per_stock = position_signal / n
        return {symbol: weight_per_stock for symbol in buy_signals}
