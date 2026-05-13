"""复合择时器。

聚合多个子择时器的信号,按"投票"或"加权平均"输出最终决策。
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum
import pandas as pd

from ..base.timing import ITimingGenerator


class SignalType(IntEnum):
    """信号类型。"""
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class Signal:
    """择时信号。"""
    symbol: str
    signal_type: SignalType
    strength: float
    price: float
    reason: str = ""
    factors: Optional[Dict[str, Any]] = None


class CompositeTiming(ITimingGenerator):
    """组合多个 ITimingGenerator 的信号。"""

    def __init__(self,
                 timings: List[ITimingGenerator],
                 mode: str = "vote"):
        if mode not in ("vote", "weighted"):
            raise ValueError(f"mode 必须是 'vote' 或 'weighted',收到: {mode}")
        if not timings:
            raise ValueError("timings 不能为空")
        self.timings = timings
        self.mode = mode

    def generate(self, factor_data: pd.DataFrame,
                 positions: List[str], cash: float, date=None) -> List[Signal]:
        bucket: dict = {}
        for timing in self.timings:
            for sig in timing.generate(factor_data, positions, cash, date):
                bucket.setdefault(sig.symbol, []).append(sig)
        if not bucket:
            return []

        result: List[Signal] = []
        for symbol, sigs in bucket.items():
            avg_price = sum(s.price for s in sigs) / len(sigs)
            avg_strength = sum(s.strength for s in sigs) / len(sigs)

            if self.mode == "vote":
                buy = sum(1 for s in sigs if s.signal_type == SignalType.BUY)
                sell = sum(1 for s in sigs if s.signal_type == SignalType.SELL)
                hold = sum(1 for s in sigs if s.signal_type == SignalType.HOLD)
                total = len(sigs)
                if buy > sell and buy > hold:
                    result.append(Signal(
                        symbol=symbol, signal_type=SignalType.BUY,
                        strength=avg_strength, price=avg_price,
                        reason=f"复合择时:多数看多({buy}/{total})",
                    ))
                elif sell > buy and sell > hold:
                    result.append(Signal(
                        symbol=symbol, signal_type=SignalType.SELL,
                        strength=avg_strength, price=avg_price,
                        reason=f"复合择时:多数看空({sell}/{total})",
                    ))
                else:
                    if symbol in positions:
                        result.append(Signal(
                            symbol=symbol, signal_type=SignalType.HOLD,
                            strength=avg_strength, price=avg_price,
                            reason=f"复合择时:中性({buy}买/{sell}卖/{hold}持)",
                        ))
            else:
                buy_w = sum(s.strength for s in sigs if s.signal_type == SignalType.BUY)
                sell_w = sum(s.strength for s in sigs if s.signal_type == SignalType.SELL)
                if buy_w > sell_w * 1.5:
                    result.append(Signal(
                        symbol=symbol, signal_type=SignalType.BUY,
                        strength=buy_w / len(sigs), price=avg_price,
                        reason=f"复合择时(加权):看多 {buy_w:.2f} > 看空 {sell_w:.2f}",
                    ))
                elif sell_w > buy_w * 1.5:
                    result.append(Signal(
                        symbol=symbol, signal_type=SignalType.SELL,
                        strength=sell_w / len(sigs), price=avg_price,
                        reason=f"复合择时(加权):看空 {sell_w:.2f} > 看多 {buy_w:.2f}",
                    ))
                elif symbol in positions:
                    result.append(Signal(
                        symbol=symbol, signal_type=SignalType.HOLD,
                        strength=avg_strength, price=avg_price,
                        reason="复合择时(加权):中性",
                    ))
        return result
