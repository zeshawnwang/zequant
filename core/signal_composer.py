"""信号组合器。

将选股信号和择时信号组合成目标持仓权重的模块。

职责:
  - 接收选股器输出的股票得分
  - 接收择时器输出的仓位系数 (0~1)
  - 组合两者，输出目标持仓权重

设计原则:
  - 择时器只决定"做多少仓位"
  - 选股器决定"持仓哪些股票"
  - SignalComposer 负责将两者组合

用法
----
    from core.signal_composer import SignalComposer

    composer = SignalComposer(
        portfolio_builder=EqualWeightBuilder(),
        risk_limits={"max_single_weight": 0.05},
    )

    target_weights = composer.compose(
        date=date,
        selector_scores={"000001.SZ": 0.9, "000002.SZ": 0.8, ...},
        timing_position=0.8,  # 80% 仓位
        cash=1_000_000,
    )
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd


class SignalComposer:
    """
    信号组合器。

    将选股信号和择时仓位系数组合成目标持仓权重。
    """

    def __init__(
        self,
        portfolio_builder=None,
        risk_limits: Optional[Dict] = None,
    ):
        """
        Args:
            portfolio_builder: 仓位分配器，负责根据得分分配权重
            risk_limits: 风控约束
                - max_single_weight: 单票最大权重
                - max_total_position: 最大总仓位
                - max_turnover: 最大换手率
        """
        self.portfolio_builder = portfolio_builder
        self.risk_limits = risk_limits or {}

    def compose(
        self,
        date,
        selector_scores: Dict[str, float],
        timing_position: float,
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        组合选股信号和择时仓位，输出目标权重。

        Args:
            date: 当前日期
            selector_scores: 选股器输出的股票得分 {symbol: score}
            timing_position: 择时仓位系数 (0~1)
            cash: 当前现金
            current_weights: 当前持仓权重（用于换手控制）

        Returns:
            目标持仓权重 {symbol: weight}
        """
        if not selector_scores or timing_position <= 0:
            return {}

        available_cash = cash * timing_position

        sorted_scores = sorted(
            selector_scores.items(), key=lambda x: x[1], reverse=True
        )

        if self.portfolio_builder:
            target_weights = self.portfolio_builder.build(
                date=date,
                scores=dict(sorted_scores),
                available_cash=available_cash,
            )
        else:
            target_weights = self._equal_weight(
                dict(sorted_scores), available_cash
            )

        target_weights = self._apply_risk_limits(
            target_weights, available_cash
        )

        if current_weights:
            target_weights = self._apply_turnover_limit(
                target_weights, current_weights
            )

        return target_weights

    def _equal_weight(
        self, scores: Dict[str, float], cash: float
    ) -> Dict[str, float]:
        """等权重分配。"""
        n = len(scores)
        if n == 0:
            return {}
        weight_per_stock = 1.0 / n
        return {symbol: weight_per_stock for symbol in scores}

    def _apply_risk_limits(
        self, weights: Dict[str, float], cash: float
    ) -> Dict[str, float]:
        """应用风控约束。"""
        max_single = self.risk_limits.get("max_single_weight", 1.0)
        max_total = self.risk_limits.get("max_total_position", 1.0)

        total_weight = sum(weights.values())

        if total_weight > max_total:
            scale = max_total / total_weight
            weights = {k: v * scale for k, v in weights.items()}
            total_weight = sum(weights.values())

        clipped_weights = {}
        for symbol, weight in weights.items():
            if weight > max_single:
                clipped_weights[symbol] = max_single
            else:
                clipped_weights[symbol] = weight

        return clipped_weights

    def _apply_turnover_limit(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """应用换手率限制。"""
        max_turnover = self.risk_limits.get("max_turnover", 1.0)

        all_symbols = set(target_weights.keys()) | set(current_weights.keys())

        turnover = sum(
            abs(target_weights.get(s, 0) - current_weights.get(s, 0))
            for s in all_symbols
        ) / 2

        if turnover > max_turnover:
            scale = max_turnover / turnover
            target_weights = {
                k: v * scale for k, v in target_weights.items()
            }

        return target_weights


class TimingSignal:
    """择时信号（可选的择时器输出封装）。"""

    def __init__(
        self,
        position: float,
        confidence: float = 1.0,
        reason: str = "",
    ):
        """
        Args:
            position: 仓位系数 (0~1)
            confidence: 信号置信度 (0~1)
            reason: 信号原因
        """
        self.position = position
        self.confidence = confidence
        self.reason = reason

    def __repr__(self):
        return (
            f"TimingSignal(position={self.position:.2f}, "
            f"confidence={self.confidence:.2f}, "
            f"reason='{self.reason}')"
        )


class SelectorScore:
    """选股得分（可选的选股器输出封装）。"""

    def __init__(
        self,
        symbol: str,
        score: float,
        rank: int,
        factors: Optional[Dict] = None,
    ):
        self.symbol = symbol
        self.score = score
        self.rank = rank
        self.factors = factors or {}

    def __repr__(self):
        return f"SelectorScore({self.symbol}, score={self.score:.3f}, rank={self.rank})"
