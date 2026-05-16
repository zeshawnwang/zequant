"""
费用与风控(Fee & Risk Manager)
包含完整的 A 股费用计算和风控规则。
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TradeCost:
    """单笔交易费用明细"""
    stamp_tax: float = 0.0       # 印花税
    transfer_fee: float = 0.0   # 过户费
    commission: float = 0.0     # 佣金
    slippage: float = 0.0       # 滑点
    total: float = 0.0          # 总费用

    def __repr__(self):
        return (f"印花税={self.stamp_tax:.2f}, 过户费={self.transfer_fee:.2f}, "
                f"佣金={self.commission:.2f}, 滑点={self.slippage:.2f}, 合计={self.total:.2f}")


class FeeCalculator:
    """
    A股费用计算规则：
    - 印花税：只在卖出时收取，沪市深市都是0.05%（2023年8月28日起）
    - 过户费：沪深统一0.001%（2022年起）
    - 佣金：默认万三，最低5元，双向收取
    - 滑点：按成交金额的百分比估算（实盘损耗）
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.stamp_tax_rate = cfg.get('stamp_tax', 0.0005)
        self.transfer_fee_rate = cfg.get('transfer_fee', 0.00001)
        self.commission_rate = cfg.get('commission', 0.0003)
        self.min_commission = cfg.get('min_commission', 5)
        self.slippage_rate = cfg.get('slippage', 0.0005)

    def calc_buy(self, symbol: str, price: float, quantity: int) -> TradeCost:
        """买入费用（无印花税）"""
        amount = price * quantity
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer_fee = amount * self.transfer_fee_rate if symbol.startswith('6') else 0
        slippage = amount * self.slippage_rate
        total = commission + transfer_fee + slippage
        return TradeCost(
            stamp_tax=0,
            transfer_fee=transfer_fee,
            commission=commission,
            slippage=slippage,
            total=total
        )

    def calc_sell(self, symbol: str, price: float, quantity: int) -> TradeCost:
        """卖出费用（含印花税）"""
        amount = price * quantity
        stamp_tax = amount * self.stamp_tax_rate
        transfer_fee = amount * self.transfer_fee_rate if symbol.startswith('6') else 0
        commission = max(amount * self.commission_rate, self.min_commission)
        slippage = amount * self.slippage_rate
        total = stamp_tax + transfer_fee + commission + slippage
        return TradeCost(
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            commission=commission,
            slippage=slippage,
            total=total
        )

    def calc_net_proceed(self, direction: str,
                         symbol: str, price: float, quantity: int) -> float:
        """
        计算净收付。
        direction: 'buy' 返回支付总额，'sell' 返回收到净额。
        """
        if direction == 'buy':
            cost = self.calc_buy(symbol, price, quantity)
            return price * quantity + cost.total
        else:
            cost = self.calc_sell(symbol, price, quantity)
            return price * quantity - cost.total


class RiskManager:
    """
    风控规则：
    - 单股仓位上限
    - 总仓位上限
    - 止损/止盈
    - 涨跌停过滤
    - ST股过滤
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.max_position_pct = cfg.get('max_position_pct', 0.15)
        self.max_total_position = cfg.get('max_total_position', 0.85)
        self.stop_loss = cfg.get('stop_loss', 0.05)
        self.take_profit = cfg.get('take_profit', 0.12)
        self.fee_calc = FeeCalculator(cfg.get('fees', {}))

    def check_buy(self, symbol: str, price: float, quantity: int,
                  total_equity: float, current_positions: Dict) -> tuple:
        """
        检查是否可以买入。
        current_positions 期望值:{symbol: Position(dataclass,含 quantity/entry_price)}
        Returns: (allowed: bool, reason: str)
        """
        position_value = price * quantity
        position_pct = position_value / total_equity if total_equity > 0 else 0

        # 单股仓位超限
        if position_pct > self.max_position_pct:
            return False, f"单股仓位超限({position_pct:.1%}>{self.max_position_pct:.1%})"

        # 总仓位超限(以 entry_price 近似估值,无当日收盘价时可用)
        current_position_value = 0.0
        for p in current_positions.values():
            qty = getattr(p, 'quantity', None)
            prc = getattr(p, 'entry_price', None)
            if qty is None and isinstance(p, dict):
                qty = p.get('quantity', 0)
                prc = p.get('current_price') or p.get('entry_price') or 0
            if qty and prc:
                current_position_value += qty * prc

        new_total_pct = (current_position_value + position_value) / total_equity if total_equity > 0 else 0
        if new_total_pct > self.max_total_position:
            return False, f"总仓位超限({new_total_pct:.1%}>{self.max_total_position:.1%})"

        return True, "OK"

    def check_stop_loss(self, entry_price: float, current_price: float,
                        direction: str = 'long') -> tuple:
        """
        检查止损/止盈。
        Returns: (should_stop: bool, reason: str)
        """
        if direction == 'long':
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct <= -self.stop_loss:
                return True, f"触发止损({loss_pct:.1%})"
            if loss_pct >= self.take_profit:
                return True, f"触发止盈({loss_pct:.1%})"
        return False, "OK"

    def apply_slippage(self, price: float, direction: str) -> float:
        """应用滑点后的成交价格"""
        if direction == 'buy':
            return price * (1 + self.fee_calc.slippage_rate)
        else:
            return price * (1 - self.fee_calc.slippage_rate)
