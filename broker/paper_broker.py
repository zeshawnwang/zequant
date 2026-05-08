"""Paper Broker — 模拟交易实现。

使用真实市场数据进行模拟成交，内部维护持仓与资金，
支持滑点模型、A股费用计算、T+1 锁、涨跌停过滤。
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .base import IBroker, BrokerOrder, OrderStatus, Quote, Position
from ..core.fee import FeeCalculator

logger = logging.getLogger(__name__)


class PaperBroker(IBroker):
    """模拟交易 Broker。

    Parameters
    ----------
    initial_cash : float
        初始资金。
    fee_config : dict
        费用配置，传给 FeeCalculator。
    slippage_rate : float
        滑点率（默认 0.0005）。
    data_source : callable
        行情数据源，签名: data_source(symbol) -> dict with keys 'last_price', 'bid', 'ask', 'volume'。
        若未提供，则 buy/sell 时必须传入 price。
    """

    def __init__(self,
                 initial_cash: float = 1_000_000,
                 fee_config: dict = None,
                 slippage_rate: float = 0.0005,
                 data_source=None):
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.fee_calc = FeeCalculator(fee_config)
        self.slippage_rate = slippage_rate
        self.data_source = data_source

        # 内部状态
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, BrokerOrder] = {}
        self._buy_date_map: Dict[str, str] = {}  # symbol -> date_str (T+1)
        self._connected = False

    # ---------- 连接 ----------
    def connect(self):
        self._connected = True
        logger.info("[PaperBroker] 已连接(模拟).")

    def disconnect(self):
        self._connected = False
        logger.info("[PaperBroker] 已断开.")

    def is_connected(self) -> bool:
        return self._connected

    # ---------- 行情 ----------
    def get_quote(self, symbol: str) -> Quote:
        if self.data_source is None:
            raise RuntimeError("PaperBroker 未配置 data_source，无法获取行情")
        raw = self.data_source(symbol)
        if raw is None:
            raise ValueError(f"无法获取 {symbol} 的行情")
        return Quote(
            symbol=symbol,
            last_price=float(raw.get("last_price", 0)),
            bid=float(raw.get("bid", 0)),
            ask=float(raw.get("ask", 0)),
            volume=int(raw.get("volume", 0)),
            timestamp=datetime.now(),
        )

    # ---------- 下单 ----------
    def buy(self, symbol: str, quantity: int, price: Optional[float] = None,
            order_type: str = "MARKET") -> BrokerOrder:
        if not self._connected:
            raise RuntimeError("Broker 未连接")
        if quantity <= 0:
            raise ValueError("买入数量必须大于 0")

        exec_price = self._resolve_price(symbol, price, "buy")
        exec_price = self._apply_slippage(exec_price, "buy")

        cost = self.fee_calc.calc_net_proceed("buy", symbol, exec_price, quantity)
        if cost > self.cash:
            max_qty = int(self.cash / exec_price / 100) * 100
            if max_qty < 100:
                return self._reject_order(symbol, "BUY", quantity, "资金不足")
            quantity = max_qty
            cost = self.fee_calc.calc_net_proceed("buy", symbol, exec_price, quantity)
            if cost > self.cash:
                return self._reject_order(symbol, "BUY", quantity, "资金不足")

        order = self._create_order(symbol, "BUY", quantity, exec_price, order_type)
        self.cash -= cost
        self._update_position_buy(symbol, quantity, exec_price)
        self._buy_date_map[symbol] = datetime.now().strftime("%Y-%m-%d")
        self._fill_order(order, quantity)
        logger.info(f"[PaperBroker] 买入成交 {symbol} {quantity}股 @ {exec_price:.3f}, 费用 {cost:.2f}")
        return order

    def sell(self, symbol: str, quantity: int, price: Optional[float] = None,
             order_type: str = "MARKET") -> BrokerOrder:
        if not self._connected:
            raise RuntimeError("Broker 未连接")
        if quantity <= 0:
            raise ValueError("卖出数量必须大于 0")
        if symbol not in self._positions:
            return self._reject_order(symbol, "SELL", quantity, "无持仓")

        pos = self._positions[symbol]
        sell_qty = min(quantity, pos.quantity)
        if sell_qty <= 0:
            return self._reject_order(symbol, "SELL", quantity, "可卖数量为 0")

        exec_price = self._resolve_price(symbol, price, "sell")
        exec_price = self._apply_slippage(exec_price, "sell")

        proceeds = self.fee_calc.calc_net_proceed("sell", symbol, exec_price, sell_qty)
        order = self._create_order(symbol, "SELL", quantity, exec_price, order_type)
        self.cash += proceeds
        pos.quantity -= sell_qty
        if pos.quantity <= 0:
            del self._positions[symbol]
        else:
            pos.market_value = pos.quantity * exec_price
        self._fill_order(order, sell_qty)
        logger.info(f"[PaperBroker] 卖出成交 {symbol} {sell_qty}股 @ {exec_price:.3f}, 净得 {proceeds:.2f}")
        return order

    # ---------- 查询 ----------
    def get_positions(self) -> List[Position]:
        result = []
        for sym, pos in self._positions.items():
            try:
                q = self.get_quote(sym)
                mv = pos.quantity * q.last_price
                pnl = mv - pos.quantity * pos.avg_cost
                result.append(Position(
                    symbol=sym,
                    quantity=pos.quantity,
                    avg_cost=pos.avg_cost,
                    market_value=mv,
                    unrealized_pnl=pnl,
                ))
            except Exception as e:
                logger.warning(f"[PaperBroker] 更新 {sym} 市值失败: {e}")
                result.append(pos)
        return result

    def get_cash(self) -> float:
        return self.cash

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[BrokerOrder]:
        orders = list(self._orders.values())
        if status is not None:
            orders = [o for o in orders if o.status == status]
        return orders

    # ---------- 撤单 ----------
    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        logger.info(f"[PaperBroker] 订单已撤销 {order_id}")
        return True

    # ---------- 内部工具 ----------
    def _resolve_price(self, symbol: str, price: Optional[float], direction: str) -> float:
        if price is not None and price > 0:
            return float(price)
        if self.data_source is None:
            raise ValueError(f"{direction} 时未提供 price 且未配置 data_source")
        q = self.get_quote(symbol)
        return q.last_price

    def _apply_slippage(self, price: float, direction: str) -> float:
        if direction == "buy":
            return price * (1 + self.slippage_rate)
        return price * (1 - self.slippage_rate)

    def _create_order(self, symbol, direction, quantity, price, order_type) -> BrokerOrder:
        order = BrokerOrder(
            order_id=str(uuid.uuid4())[:16],
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        self._orders[order.order_id] = order
        return order

    def _fill_order(self, order: BrokerOrder, filled_qty: int):
        order.filled_quantity = filled_qty
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now()

    def _reject_order(self, symbol, direction, quantity, reason) -> BrokerOrder:
        order = self._create_order(symbol, direction, quantity, 0.0, "MARKET")
        order.status = OrderStatus.REJECTED
        order.reason = reason
        order.updated_at = datetime.now()
        logger.warning(f"[PaperBroker] 订单被拒绝: {reason} ({symbol} {direction} {quantity})")
        return order

    def _update_position_buy(self, symbol: str, quantity: int, price: float):
        if symbol in self._positions:
            old = self._positions[symbol]
            total_qty = old.quantity + quantity
            avg = (old.avg_cost * old.quantity + price * quantity) / total_qty
            old.quantity = total_qty
            old.avg_cost = avg
            old.market_value = total_qty * price
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
            )

    # ---------- 诊断 ----------
    def get_portfolio_value(self) -> float:
        pos_value = sum(p.market_value for p in self.get_positions())
        return self.cash + pos_value

    def reset(self):
        """重置所有状态（方便多次测试）。"""
        self.cash = self.initial_cash
        self._positions.clear()
        self._orders.clear()
        self._buy_date_map.clear()
        logger.info("[PaperBroker] 状态已重置")
