"""
Backtest Engine
事件驱动回测,支持费用计算、滑点、止损止盈。

修复说明:
- 旧版 _check_stops 是空实现,止损止盈不生效。
- 旧版胜率/profit_factor 用「单笔成交价 vs 初始资金」比较,口径错误。
  现按 entry/exit 配对(FIFO)计算每笔实现盈亏。
"""
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List
from dataclasses import dataclass, field

from .strategy import Order, Position, QuantStrategy, SignalType
from .fee import FeeCalculator, RiskManager


@dataclass
class Trade:
    """成交记录"""
    date: str
    symbol: str
    direction: str
    price: float
    quantity: int
    commission: float


@dataclass
class BacktestReport:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.DataFrame
    trades: List[Trade] = field(default_factory=list)


class BacktestEngine:
    """事件驱动回测引擎。"""

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 fee_config: dict = None,
                 risk_config: dict = None):
        self.initial_capital = initial_capital
        self.fee_calc = FeeCalculator(fee_config)
        # RiskManager 内部还会读 fees 子键,这里把 fee 也传进去
        rcfg = dict(risk_config or {})
        if fee_config and "fees" not in rcfg:
            rcfg["fees"] = fee_config
        self.risk_mgr = RiskManager(rcfg)
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.trades: List[Trade] = []
        self.daily_values: List[dict] = []

    def run(self,
            strategy: QuantStrategy,
            factor_data: pd.DataFrame,
            start_date: str,
            end_date: str,
            rebalance_freq: str = '1d') -> BacktestReport:
        if 'date' not in factor_data.columns:
            raise ValueError("factor_data must have 'date' column")

        df = factor_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[
            (df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date))
        ]
        if df.empty:
            print("回测区间内无数据")
            return self._generate_report()

        dates = sorted(df['date'].unique())

        for current_date in dates:
            day_slice = df[df['date'] <= current_date]
            today_only = df[df['date'] == current_date]

            # Step 1: 先按当日收盘价检查止损止盈
            self._check_stops(today_only, str(pd.Timestamp(current_date).date()))

            # Step 2: 让策略生成订单
            orders = strategy.generate_orders(
                day_slice, self.positions, self.cash, current_date
            )

            for order in orders:
                self._execute_order(order, str(pd.Timestamp(current_date).date()))

            # Step 3: 记录每日净值
            portfolio_value = self._calc_portfolio_value(today_only)
            self.daily_values.append({
                'date': current_date,
                'cash': self.cash,
                'portfolio_value': portfolio_value,
                'total_value': self.cash + portfolio_value,
            })

        return self._generate_report()

    def _execute_order(self, order: Order, date: str):
        if order.quantity <= 0 or order.price <= 0:
            return

        exec_price = self.risk_mgr.apply_slippage(
            order.price, order.direction.lower()
        )
        if exec_price <= 0:
            return

        if order.direction == 'BUY':
            cost = self.fee_calc.calc_net_proceed('buy', order.symbol, exec_price, order.quantity)
            qty = order.quantity
            if cost > self.cash:
                # 现金不足,按可买数量(向下取整到 100 股)再核
                max_qty = int(self.cash / exec_price / 100) * 100
                if max_qty < 100:
                    return
                qty = max_qty
                cost = self.fee_calc.calc_net_proceed('buy', order.symbol, exec_price, qty)
                if cost > self.cash:
                    return

            self.cash -= cost
            # 已持仓加仓:用加权均价更新
            if order.symbol in self.positions:
                old = self.positions[order.symbol]
                total_qty = old.quantity + qty
                avg_price = (old.entry_price * old.quantity + exec_price * qty) / total_qty
                old.quantity = total_qty
                old.entry_price = avg_price
                old.stop_loss = avg_price * (1 - self.risk_mgr.stop_loss)
                old.take_profit = avg_price * (1 + self.risk_mgr.take_profit)
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=qty,
                    entry_price=exec_price,
                    entry_date=date,
                    stop_loss=exec_price * (1 - self.risk_mgr.stop_loss),
                    take_profit=exec_price * (1 + self.risk_mgr.take_profit),
                )
            self.trades.append(Trade(
                date=date, symbol=order.symbol,
                direction='BUY', price=exec_price, quantity=qty,
                commission=self.fee_calc.calc_buy(order.symbol, exec_price, qty).total,
            ))

        elif order.direction == 'SELL':
            if order.symbol not in self.positions:
                return
            pos = self.positions[order.symbol]
            sell_qty = min(order.quantity, pos.quantity)
            if sell_qty <= 0:
                return
            proceeds = self.fee_calc.calc_net_proceed('sell', order.symbol, exec_price, sell_qty)
            self.cash += proceeds
            pos.quantity -= sell_qty
            if pos.quantity <= 0:
                del self.positions[order.symbol]
            self.trades.append(Trade(
                date=date, symbol=order.symbol,
                direction='SELL', price=exec_price, quantity=sell_qty,
                commission=self.fee_calc.calc_sell(order.symbol, exec_price, sell_qty).total,
            ))

    def _check_stops(self, today_only: pd.DataFrame, date: str):
        """根据当日 close 对持仓检查止损/止盈,触发即按当日 close 平仓。"""
        if today_only is None or today_only.empty or not self.positions:
            return
        price_map = dict(zip(today_only['symbol'], today_only['close']))
        to_sell = []
        for symbol, pos in self.positions.items():
            cp = price_map.get(symbol)
            if cp is None or pd.isna(cp):
                continue
            cp = float(cp)
            should_stop, reason = self.risk_mgr.check_stop_loss(
                pos.entry_price, cp, direction='long'
            )
            if should_stop:
                to_sell.append((symbol, cp, reason))
        for symbol, cp, reason in to_sell:
            self._execute_order(
                Order(symbol=symbol, direction='SELL',
                      quantity=self.positions[symbol].quantity,
                      price=cp, reason=reason),
                date,
            )

    def _calc_portfolio_value(self, today_only: pd.DataFrame) -> float:
        if today_only is None or today_only.empty:
            return sum(p.quantity * p.entry_price for p in self.positions.values())
        price_map = dict(zip(today_only['symbol'], today_only['close']))
        total = 0.0
        for symbol, pos in self.positions.items():
            cp = price_map.get(symbol, pos.entry_price)
            if cp is None or pd.isna(cp):
                cp = pos.entry_price
            total += pos.quantity * float(cp)
        return total

    def _generate_report(self) -> BacktestReport:
        equity = pd.DataFrame(self.daily_values)
        if equity.empty:
            return BacktestReport(0, 0, 0, 0, 0, 0, 0, equity)

        total_value = equity['total_value'].astype(float)
        returns = total_value.pct_change().dropna()

        final_value = total_value.iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        n_days = len(equity)
        n_years = n_days / 252
        annualized_return = (
            (final_value / self.initial_capital) ** (1 / n_years) - 1
            if n_years > 0 and final_value > 0 else 0
        )

        cummax = total_value.cummax()
        drawdown = (total_value - cummax) / cummax
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        sharpe_ratio = (
            float(returns.mean() / returns.std() * np.sqrt(252))
            if returns.std() > 0 else 0.0
        )

        # ===== FIFO 配对计算实现盈亏 =====
        win_rate, profit_factor = self._calc_pnl_stats()

        return BacktestReport(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            equity_curve=equity,
            trades=self.trades,
        )

    def _calc_pnl_stats(self):
        """按 FIFO 配对每只股票的 BUY/SELL,计算每笔实现盈亏。"""
        buys: Dict[str, deque] = defaultdict(deque)
        realized = []
        for t in self.trades:
            if t.direction == 'BUY':
                buys[t.symbol].append([t.price, t.quantity])
            else:  # SELL
                qty_to_close = t.quantity
                while qty_to_close > 0 and buys[t.symbol]:
                    open_price, open_qty = buys[t.symbol][0]
                    closed = min(open_qty, qty_to_close)
                    pnl = (t.price - open_price) * closed
                    realized.append(pnl)
                    qty_to_close -= closed
                    if closed >= open_qty:
                        buys[t.symbol].popleft()
                    else:
                        buys[t.symbol][0][1] = open_qty - closed

        if not realized:
            return 0.0, 0.0
        wins = [p for p in realized if p > 0]
        losses = [p for p in realized if p < 0]
        win_rate = len(wins) / len(realized)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        # 处理 inf 防止后续打印报错
        if profit_factor == float('inf'):
            profit_factor = 999.0
        return float(win_rate), float(profit_factor)