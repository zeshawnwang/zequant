"""复合择时器。

聚合多个子择时器的信号,按"投票"或"加权平均"输出最终决策。

模式:
  - "vote":     按 BUY/SELL/HOLD 三类信号的票数取多数(平票时取 HOLD)
  - "weighted": 按 strength 加权平均,正向多数 → BUY、负向多数 → SELL,
                两者均未过半 → HOLD

关键修复(对比上一版):上一版投票模式下 buy_count > sell_count 时只发 HOLD,
导致复合择时永远不建仓。本版多数票 BUY 也会发 BUY 信号。
"""
from __future__ import annotations
from typing import List
import pandas as pd

from core.strategy import Signal, SignalType
from .base import ITimingGenerator


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
                 positions: List[str], cash: float) -> List[Signal]:
        # 1) 收集所有子择时器的信号,按 symbol 分桶
        bucket: dict = {}
        for timing in self.timings:
            for sig in timing.generate(factor_data, positions, cash):
                bucket.setdefault(sig.symbol, []).append(sig)
        if not bucket:
            return []

        # 2) 聚合
        result: List[Signal] = []
        for symbol, sigs in bucket.items():
            avg_price = sum(s.price for s in sigs) / len(sigs)
            avg_strength = sum(s.strength for s in sigs) / len(sigs)

            if self.mode == "vote":
                buy = sum(1 for s in sigs if s.signal_type == SignalType.BUY)
                sell = sum(1 for s in sigs if s.signal_type == SignalType.SELL)
                hold = sum(1 for s in sigs if s.signal_type == SignalType.HOLD)
                # 多数投票:BUY/SELL 至少要过半票数才生效,否则视为 HOLD
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
                    # 平票或多数 HOLD,持仓继续持有,空仓不动作
                    if symbol in positions:
                        result.append(Signal(
                            symbol=symbol, signal_type=SignalType.HOLD,
                            strength=avg_strength, price=avg_price,
                            reason=f"复合择时:中性({buy}买/{sell}卖/{hold}持)",
                        ))
            else:  # weighted
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