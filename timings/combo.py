"""
复合择时器
组合多个择时信号。
"""
from typing import List
import pandas as pd
from .trend import ITimingGenerator
from core.strategy import Signal, SignalType


class CompositeTiming(ITimingGenerator):
    """
    复合择时器。
    汇总多个择时器的信号，按多数表决或加权评分。
    """

    def __init__(self,
                 timings: List[ITimingGenerator],
                 mode: str = 'vote'):  # 'vote' or 'weighted'
        self.timings = timings
        self.mode = mode

    def generate(self, factor_data: pd.DataFrame,
                positions: List[str], cash: float) -> List[Signal]:
        all_signal_lists = []
        for timing in self.timings:
            signals = timing.generate(factor_data, positions, cash)
            all_signal_lists.append(signals)

        # 按股票聚合信号
        symbol_signals = {}
        for signals in all_signal_lists:
            for sig in signals:
                if sig.symbol not in symbol_signals:
                    symbol_signals[sig.symbol] = []
                symbol_signals[sig.symbol].append(sig)

        # 汇总
        result = []
        for symbol, sigs in symbol_signals.items():
            if not sigs:
                continue

            if self.mode == 'vote':
                buy_count = sum(1 for s in sigs if s.signal_type == SignalType.BUY)
                sell_count = sum(1 for s in sigs if s.signal_type == SignalType.SELL)
                avg_price = sum(s.price for s in sigs) / len(sigs)
                avg_strength = sum(s.strength for s in sigs) / len(sigs)

                if sell_count > buy_count:
                    result.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=avg_strength,
                        price=avg_price,
                        reason=f"多数择时卖出({sell_count}/{len(sigs)})"
                    ))
                elif buy_count > sell_count:
                    result.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.HOLD,
                        strength=avg_strength,
                        price=avg_price,
                        reason=f"多数择时持有({buy_count}/{len(sigs)})"
                    ))
            else:
                # weighted: 取平均信号强度
                avg_strength = sum(s.strength for s in sigs) / len(sigs)
                avg_price = sum(s.price for s in sigs) / len(sigs)
                result.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    strength=avg_strength,
                    price=avg_price,
                    reason=f"复合择时({len(sigs)}个指标)"
                ))

        return result
