"""信号流驱动的策略基类。

核心设计理念：
    - 择时器输出仓位系数 (0~1)
    - 选股器输出股票得分
    - 信号组合器将两者组合成目标权重
    - 风控层对权重进行约束

执行流程：
    1. Selector.get_scores(date) → 选股得分
    2. PositionSizer.get_position(date) → 仓位系数
    3. Composer.compose(scores, position, cash) → 目标权重
    4. RiskManager.apply(weights) → 最终权重
    5. 生成交易清单

用法
----
    from core.strategy import SignalStrategy
    from core.signals import LayeredComposer
    from core.signals.position import TrendPositionSizer
    from screening import FactorRankSelector

    strategy = SignalStrategy(
        name="MomentumStrategy",
        selector=FactorRankSelector(factor_name="momentum_20"),
        position_sizer=TrendPositionSizer(),
        composer=LayeredComposer(top_n=30),
        risk_manager=RiskManager(),
    )
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class TargetPosition:
    """目标持仓。"""
    symbol: str
    target_weight: float
    target_value: float
    current_weight: float = 0.0
    current_value: float = 0.0
    signal_type: str = "hold"

    @property
    def order_value(self) -> float:
        """需要交易的金额（正=买入，负=卖出）。"""
        return self.target_value - self.current_value


@dataclass
class StrategySignal:
    """策略信号。"""
    date: Any
    position_signal: float
    selector_scores: Dict[str, float]
    target_weights: Dict[str, float]
    orders: List[TargetPosition]


class IStrategy(ABC):
    """策略接口。"""

    @abstractmethod
    def generate_orders(
        self,
        date: Any,
        market_data: pd.DataFrame,
        cash: float,
        positions: Dict[str, float],
    ) -> List[TargetPosition]:
        """
        生成交易清单。

        Args:
            date: 当前日期
            market_data: 市场数据
            cash: 当前现金
            positions: 当前持仓 {symbol: weight}

        Returns:
            目标持仓列表
        """
        pass

    @abstractmethod
    def get_signal(self, date: Any, market_data: pd.DataFrame) -> StrategySignal:
        """获取策略信号。"""
        pass


class SignalStrategy(IStrategy):
    """
    信号流驱动的策略。

    组件：
        - selector: 选股器
        - position_sizer: 仓位管理器
        - composer: 信号组合器
        - risk_manager: 风控管理器
    """

    def __init__(
        self,
        name: str,
        selector=None,
        position_sizer=None,
        composer=None,
        risk_manager=None,
        top_n: int = 30,
        min_position: float = 0.001,
    ):
        """
        Args:
            name: 策略名称
            selector: 选股器（ISelector 接口）
            position_sizer: 仓位管理器（IPositionSizer 接口）
            composer: 信号组合器（IComposer 接口）
            risk_manager: 风控管理器（IRiskManager 接口）
            top_n: 最大持仓数量
            min_position: 最小持仓权重（低于此值不持仓）
        """
        self.name = name
        self.selector = selector
        self.position_sizer = position_sizer
        self.composer = composer
        self.risk_manager = risk_manager
        self.top_n = top_n
        self.min_position = min_position

    def get_signal(
        self,
        date: Any,
        market_data: pd.DataFrame,
        cash: float,
        positions: Dict[str, float],
    ) -> StrategySignal:
        """
        获取策略信号。

        Returns:
            StrategySignal: 包含仓位信号、选股得分、目标权重的信号对象
        """
        selector_scores = self._get_selector_scores(date, market_data)

        position_signal = self._get_position_signal(date, market_data)

        current_weights = positions

        target_weights = self._compose_weights(
            selector_scores, position_signal, cash, current_weights
        )

        return StrategySignal(
            date=date,
            position_signal=position_signal,
            selector_scores=selector_scores,
            target_weights=target_weights,
            orders=[],
        )

    def generate_orders(
        self,
        date: Any,
        market_data: pd.DataFrame,
        cash: float,
        positions: Dict[str, float],
    ) -> List[TargetPosition]:
        """
        生成交易清单。

        Args:
            date: 当前日期
            market_data: 市场数据
            cash: 当前现金
            positions: 当前持仓 {symbol: weight}

        Returns:
            目标持仓列表
        """
        signal = self.get_signal(date, market_data, cash, positions)

        orders = self._generate_orders_from_weights(
            date, signal.target_weights, cash, positions
        )

        signal.orders = orders

        return orders

    def _get_selector_scores(
        self, date: Any, market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """获取选股得分。"""
        if self.selector is None:
            return {}

        if hasattr(self.selector, "select"):
            scores = self.selector.select(date, market_data)
        elif hasattr(self.selector, "get_scores"):
            scores = self.selector.get_scores(date, market_data)
        else:
            scores = {}

        return scores

    def _get_position_signal(
        self, date: Any, market_data: pd.DataFrame
    ) -> float:
        """获取仓位信号。"""
        if self.position_sizer is None:
            return 1.0

        if hasattr(self.position_sizer, "get_position"):
            position = self.position_sizer.get_position(date, market_data)
        elif hasattr(self.position_sizer, "generate"):
            signals = self.position_sizer.generate(market_data, [], 0, date)
            position = self._signals_to_position(signals)
        else:
            position = 1.0

        return max(0.0, min(1.0, position))

    def _signals_to_position(self, signals) -> float:
        """将择时信号转换为仓位系数。"""
        if not signals:
            return 1.0

        buy_count = sum(1 for s in signals if s.signal_type.value == "buy")
        sell_count = sum(1 for s in signals if s.signal_type.value == "sell")

        if buy_count > sell_count:
            return 0.8
        elif sell_count > buy_count:
            return 0.2
        else:
            return 1.0

    def _compose_weights(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """组合信号生成权重。"""
        if self.composer is None:
            return self._default_compose(
                selector_scores, position_signal, cash, current_weights
            )

        weights = self.composer.compose(
            selector_scores=selector_scores,
            position_signal=position_signal,
            cash=cash,
            current_weights=current_weights,
        )

        return weights

    def _default_compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """默认的组合逻辑（等权重）。"""
        if not selector_scores:
            return {}

        sorted_scores = sorted(
            selector_scores.items(), key=lambda x: x[1], reverse=True
        )

        top_symbols = [s for s, _ in sorted_scores[: self.top_n]]

        n = len(top_symbols)
        if n == 0:
            return {}

        weight_per_stock = position_signal / n

        weights = {symbol: weight_per_stock for symbol in top_symbols}

        if self.risk_manager is not None:
            weights = self.risk_manager.apply(weights, cash, current_weights)

        return weights

    def _generate_orders_from_weights(
        self,
        date: Any,
        target_weights: Dict[str, float],
        cash: float,
        current_positions: Dict[str, float],
    ) -> List[TargetPosition]:
        """根据目标权重生成交易清单。"""
        total_value = cash + sum(
            current_positions.get(s, 0) * cash for s in current_positions
        )

        orders = []

        all_symbols = set(target_weights.keys()) | set(current_positions.keys())

        for symbol in all_symbols:
            target_weight = target_weights.get(symbol, 0)
            current_weight = current_positions.get(symbol, 0)

            if abs(target_weight - current_weight) < self.min_position:
                continue

            target_value = target_weight * total_value
            current_value = current_weight * total_value

            order = TargetPosition(
                symbol=symbol,
                target_weight=target_weight,
                target_value=target_value,
                current_weight=current_weight,
                current_value=current_value,
                signal_type="buy" if target_weight > current_weight else "sell",
            )

            orders.append(order)

        return orders

    def get_description(self) -> str:
        """获取策略描述。"""
        parts = [f"策略名称: {self.name}"]

        if self.selector:
            parts.append(f"选股器: {self.selector.__class__.__name__}")
        if self.position_sizer:
            parts.append(f"仓位管理: {self.position_sizer.__class__.__name__}")
        if self.composer:
            parts.append(f"信号组合: {self.composer.__class__.__name__}")
        if self.risk_manager:
            parts.append(f"风控管理: {self.risk_manager.__class__.__name__}")

        parts.append(f"最大持仓: {self.top_n}")
        parts.append(f"最小持仓权重: {self.min_position}")

        return "\n".join(parts)


class CompositeStrategy(IStrategy):
    """组合策略：组合多个子策略。"""

    def __init__(
        self,
        name: str,
        strategies: List[IStrategy],
        weights: Optional[List[float]] = None,
        mode: str = "average",
    ):
        """
        Args:
            name: 策略名称
            strategies: 子策略列表
            weights: 各策略权重
            mode: 组合模式 ("average", "vote", "weighted")
        """
        self.name = name
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        self.mode = mode

        if len(self.strategies) != len(self.weights):
            raise ValueError("strategies and weights must have same length")

    def generate_orders(
        self,
        date: Any,
        market_data: pd.DataFrame,
        cash: float,
        positions: Dict[str, float],
    ) -> List[TargetPosition]:
        """组合多个策略的订单。"""
        all_orders = []

        for strategy, weight in zip(self.strategies, self.weights):
            orders = strategy.generate_orders(date, market_data, cash * weight, positions)
            all_orders.extend(orders)

        return self._aggregate_orders(all_orders)

    def _aggregate_orders(self, orders: List[TargetPosition]) -> List[TargetPosition]:
        """聚合多个策略的订单。"""
        symbol_orders: Dict[str, List[TargetPosition]] = {}

        for order in orders:
            if order.symbol not in symbol_orders:
                symbol_orders[order.symbol] = []
            symbol_orders[order.symbol].append(order)

        aggregated = []

        for symbol, order_list in symbol_orders.items():
            total_order_value = sum(o.order_value for o in order_list)
            avg_weight = sum(o.target_weight for o in order_list) / len(order_list)

            first_order = order_list[0]

            aggregated.append(
                TargetPosition(
                    symbol=symbol,
                    target_weight=avg_weight,
                    target_value=first_order.target_value,
                    current_weight=first_order.current_weight,
                    current_value=first_order.current_value,
                    signal_type="buy" if total_order_value > 0 else "sell",
                )
            )

        return aggregated

    def get_signal(
        self, date: Any, market_data: pd.DataFrame, cash: float, positions: Dict[str, float]
    ) -> StrategySignal:
        """获取组合策略信号。"""
        signals = []

        for strategy, weight in zip(self.strategies, self.weights):
            signal = strategy.get_signal(
                date, market_data, cash * weight, positions
            )
            signals.append(signal)

        avg_position = sum(s.position_signal for s in signals) / len(signals)

        all_scores: Dict[str, List[float]] = {}
        for signal in signals:
            for symbol, score in signal.selector_scores.items():
                if symbol not in all_scores:
                    all_scores[symbol] = []
                all_scores[symbol].append(score)

        avg_scores = {
            symbol: sum(scores) / len(scores)
            for symbol, scores in all_scores.items()
        }

        return StrategySignal(
            date=date,
            position_signal=avg_position,
            selector_scores=avg_scores,
            target_weights={},
            orders=[],
        )
