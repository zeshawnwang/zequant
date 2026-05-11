"""信号组合器模块。

提供多种信号组合方式和风控约束，支持灵活组合。

组合方式
--------
    DirectComposer    : 直接相乘 (推荐)
    VoteComposer     : 投票
    WeightedComposer : 加权平均

约束类型
--------
    MaxSingleWeightConstraint      : 单票权重上限
    MaxTotalPositionConstraint    : 总仓位上限
    TurnoverConstraint            : 换手率约束
    IndustryConstraint            : 行业偏离约束
    LiquidityConstraint           : 流动性约束

用法
----
    from core.signals.composer import (
        DirectComposer, VoteComposer, WeightedComposer,
        MaxSingleWeightConstraint, MaxTotalPositionConstraint,
    )

    # 方式1：直接相乘
    composer = DirectComposer(constraints=[
        MaxSingleWeightConstraint(max_weight=0.05),
        MaxTotalPositionConstraint(max_position=0.9),
    ])

    # 方式2：投票
    composer = VoteComposer(
        signals=[selector_scores, timing_signals],
        mode="majority",
    )

    # 方式3：加权
    composer = WeightedComposer(
        weights={"selector": 0.7, "timing": 0.3},
    )
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd


class IComposer(ABC):
    """信号组合器抽象基类。"""

    @abstractmethod
    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """组合选股信号和仓位信号，输出目标权重。"""
        pass

    def apply_constraints(
        self,
        weights: Dict[str, float],
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """应用风控约束（可被子类重写）。"""
        for constraint in self.constraints:
            weights = constraint.apply(weights, cash, current_weights)
        return weights

    def add_constraint(self, constraint: "IConstraint"):
        """添加风控约束。"""
        self.constraints.append(constraint)

    def __init__(self, constraints: Optional[List["IConstraint"]] = None):
        self.constraints = constraints or []


class DirectComposer(IComposer):
    """
    直接相乘组合器。

    逻辑：目标权重 = 选股得分归一化 × 仓位系数

    适用场景：
        - 选股得分已经是 0~1 范围的概率/排名
        - 仓位系数是 0~1 的仓位建议
    """

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if not selector_scores or position_signal <= 0:
            return {}

        sorted_scores = sorted(
            selector_scores.items(), key=lambda x: x[1], reverse=True
        )

        total_score = sum(score for _, score in sorted_scores)
        if total_score <= 0:
            return {}

        normalized_weights = {
            symbol: score / total_score * position_signal
            for symbol, score in sorted_scores
        }

        return self.apply_constraints(
            normalized_weights, cash, current_weights
        )


class VoteComposer(IComposer):
    """
    投票组合器。

    逻辑：按投票结果决定是否持有

    模式：
        - "unanimous" : 全票通过
        - "majority"  : 多数通过 (>50%)
        - "at_least_one" : 至少一票通过
    """

    def __init__(
        self,
        signals: List[Dict[str, float]],
        mode: str = "majority",
        constraints: Optional[List["IConstraint"]] = None,
    ):
        super().__init__(constraints)
        self.signals = signals
        self.mode = mode

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if not selector_scores or position_signal <= 0:
            return {}

        all_symbols = set(selector_scores.keys())

        for signal in self.signals:
            all_symbols |= set(signal.keys())

        vote_counts: Dict[str, int] = {s: 0 for s in all_symbols}

        for signal in self.signals:
            threshold = self._get_threshold(signal)
            for symbol, value in signal.items():
                if value >= threshold:
                    vote_counts[symbol] += 1

        threshold_votes = self._get_vote_threshold()

        selected_symbols = [
            s for s, count in vote_counts.items() if count >= threshold_votes
        ]

        weight = position_signal / len(selected_symbols) if selected_symbols else 0

        weights = {s: weight for s in selected_symbols}

        return self.apply_constraints(weights, cash, current_weights)

    def _get_threshold(self, signal: Dict[str, float]) -> float:
        if self.mode == "unanimous":
            return max(signal.values()) * 0.9
        elif self.mode == "majority":
            return sum(signal.values()) / len(signal) / 2
        else:
            return min(signal.values()) * 1.1

    def _get_vote_threshold(self) -> int:
        n = len(self.signals)
        if self.mode == "unanimous":
            return n
        elif self.mode == "majority":
            return (n + 1) // 2
        else:
            return 1


class WeightedComposer(IComposer):
    """
    加权平均组合器。

    逻辑：目标权重 = Σ(权重_i × 信号_i) × 仓位系数

    适用场景：
        - 多信号需要按重要性加权
        - 需要灵活调整各信号权重
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        constraints: Optional[List["IConstraint"]] = None,
    ):
        super().__init__(constraints)
        self.weights = weights or {"selector": 0.7, "timing": 0.3}

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if not selector_scores or position_signal <= 0:
            return {}

        all_symbols = set(selector_scores.keys())

        timing_scores = kwargs.get("timing_scores", {})
        all_symbols |= set(timing_scores.keys())

        combined_scores: Dict[str, float] = {}
        for symbol in all_symbols:
            selector_score = selector_scores.get(symbol, 0)
            timing_score = timing_scores.get(symbol, 0)

            combined_score = (
                self.weights.get("selector", 0.5) * selector_score
                + self.weights.get("timing", 0.5) * timing_score
            )
            combined_scores[symbol] = combined_score

        sorted_scores = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        total_score = sum(score for _, score in sorted_scores)
        if total_score <= 0:
            return {}

        normalized_weights = {
            symbol: score / total_score * position_signal
            for symbol, score in sorted_scores
        }

        return self.apply_constraints(
            normalized_weights, cash, current_weights
        )


class LayeredComposer(IComposer):
    """
    分层组合器。

    逻辑：
        1. 先用择时仓位缩放总风险预算
        2. 再用选股得分分配个股权重
        3. 最后应用风控约束

    这是最符合直觉的组合方式。
    """

    def __init__(
        self,
        top_n: int = 30,
        constraints: Optional[List["IConstraint"]] = None,
    ):
        super().__init__(constraints)
        self.top_n = top_n

    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if not selector_scores or position_signal <= 0:
            return {}

        available_cash = cash * position_signal

        sorted_scores = sorted(
            selector_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_symbols = [s for s, _ in sorted_scores[: self.top_n]]

        n = len(top_symbols)
        if n == 0:
            return {}

        weight_per_stock = 1.0 / n
        weights = {symbol: weight_per_stock for symbol in top_symbols}

        return self.apply_constraints(weights, available_cash, current_weights)


# ============================================================================
# 风控约束
# ============================================================================


class IConstraint(ABC):
    """风控约束抽象基类。"""

    @abstractmethod
    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """应用约束并返回调整后的权重。"""
        pass


class MaxSingleWeightConstraint(IConstraint):
    """单票权重上限约束。"""

    def __init__(self, max_weight: float = 0.05):
        self.max_weight = max_weight

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        return {
            symbol: min(weight, self.max_weight) for symbol, weight in weights.items()
        }


class MaxTotalPositionConstraint(IConstraint):
    """总仓位上限约束。"""

    def __init__(self, max_position: float = 0.9):
        self.max_position = max_position

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        total_weight = sum(weights.values())
        if total_weight > self.max_position:
            scale = self.max_position / total_weight
            return {k: v * scale for k, v in weights.items()}
        return weights


class TurnoverConstraint(IConstraint):
    """换手率约束。"""

    def __init__(self, max_turnover: float = 0.3):
        self.max_turnover = max_turnover

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if not current_weights:
            return weights

        all_symbols = set(weights.keys()) | set(current_weights.keys())

        turnover = sum(
            abs(weights.get(s, 0) - current_weights.get(s, 0)) for s in all_symbols
        ) / 2

        if turnover > self.max_turnover:
            scale = self.max_turnover / turnover
            return {k: v * scale for k, v in weights.items()}

        return weights


class MinPositionConstraint(IConstraint):
    """最小仓位约束（低于此值的股票剔除）。"""

    def __init__(self, min_weight: float = 0.001):
        self.min_weight = min_weight

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        return {k: v for k, v in weights.items() if v >= self.min_weight}


class ReserveCashConstraint(IConstraint):
    """预留现金约束。"""

    def __init__(self, reserve_ratio: float = 0.1):
        self.reserve_ratio = reserve_ratio

    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        total_weight = sum(weights.values())
        max_weight = 1.0 - self.reserve_ratio
        if total_weight > max_weight:
            scale = max_weight / total_weight
            return {k: v * scale for k, v in weights.items()}
        return weights
