"""
Backtest Engine
基于简单事件驱动回测，支持费用计算和风控。
"""
import pandas as pd
import numpy as np
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
    """回测报告"""
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
    """
    事件驱动回测引擎。
    步骤：
    1. 按日期遍历
    2. 获取当日因子数据
    3. 执行策略生成订单
    4. 模拟撮合（含滑点）
    5. 更新持仓
    6. 记录净值
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 fee_config: dict = None,
                 risk_config: dict = None):
        self.initial_capital = initial_capital
        self.fee_calc = FeeCalculator(fee_config)
        self.risk_mgr = RiskManager(risk_config)
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
        """
        运行回测。
        """
        # 过滤日期范围
        if 'date' not in factor_data.columns:
            raise ValueError("factor_data must have 'date' column")

        df = factor_data[
            (factor_data['date'] >= pd.to_datetime(start_date)) &
            (factor_data['date'] <= pd.to_datetime(end_date))
        ]

        dates = sorted(df['date'].unique())

        for i, current_date in enumerate(dates):
            # 获取当日因子数据
            day_data = df[df['date'] <= current_date]

            # 生成订单
            orders = strategy.generate_orders(
                day_data,
                self.positions,
                self.cash,
                current_date
            )

            # 执行订单
            for order in orders:
                self._execute_order(order, str(current_date.date()))

            # 更新止损/止盈
            self._check_stops(current_date)

            # 记录每日净值
            portfolio_value = self._calc_portfolio_value(day_data)
            self.daily_values.append({
                'date': current_date,
                'cash': self.cash,
                'portfolio_value': portfolio_value,
                'total_value': self.cash + portfolio_value
            })

        return self._generate_report()

    def _execute_order(self, order: Order, date: str):
        """执行订单（模拟撮合）"""
        if order.quantity <= 0:
            return

        # 应用滑点
        exec_price = self.risk_mgr.apply_slippage(
            order.price, order.direction.lower()
        )
        if exec_price <= 0:
            return

        if order.direction == 'BUY':
            # 计算费用
            cost = self.fee_calc.calc_net_proceed('buy', order.symbol, exec_price, order.quantity)
            if cost > self.cash:
                # 钱不够，按最大可买量执行
                max_qty = int(self.cash / exec_price / 100) * 100
                if max_qty < 100:
                    return
                exec_price = self.risk_mgr.apply_slippage(order.price, 'buy')
                cost = self.fee_calc.calc_net_proceed('buy', order.symbol, exec_price, max_qty)
                order.quantity = max_qty

            self.cash -= cost
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                entry_price=exec_price,
                entry_date=date,
                stop_loss=exec_price * (1 - self.risk_mgr.stop_loss),
                take_profit=exec_price * (1 + self.risk_mgr.take_profit)
            )
            self.trades.append(Trade(
                date=date, symbol=order.symbol,
                direction='BUY', price=exec_price,
                quantity=order.quantity,
                commission=self.fee_calc.calc_buy(order.symbol, exec_price, order.quantity).total
            ))

        elif order.direction == 'SELL':
            if order.symbol not in self.positions:
                return
            pos = self.positions[order.symbol]
            sell_qty = min(order.quantity, pos.quantity)
            proceeds = self.fee_calc.calc_net_proceed('sell', order.symbol, exec_price, sell_qty)
            self.cash += proceeds
            pos.quantity -= sell_qty
            if pos.quantity <= 0:
                del self.positions[order.symbol]
            self.trades.append(Trade(
                date=date, symbol=order.symbol,
                direction='SELL', price=exec_price,
                quantity=sell_qty,
                commission=self.fee_calc.calc_sell(order.symbol, exec_price, sell_qty).total
            ))

    def _check_stops(self, date):
        """检查止损/止盈"""
        to_sell = []
        for symbol, pos in self.positions.items():
            # 获取最新收盘价
            # 简化版：跳过，实际从当日数据获取
            pass

    def _calc_portfolio_value(self, day_data: pd.DataFrame) -> float:
        total = 0
        for symbol, pos in self.positions.items():
            stock_df = day_data[day_data['symbol'] == symbol]
            if len(stock_df) > 0:
                price = stock_df['close'].iloc[-1]
                total += pos.quantity * price
        return total

    def _generate_report(self) -> BacktestReport:
        equity = pd.DataFrame(self.daily_values)
        if equity.empty:
            return BacktestReport(0, 0, 0, 0, 0, 0, 0, equity)

        total_value = equity['total_value']
        returns = total_value.pct_change().dropna()

        # 计算收益
        final_value = total_value.iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 年化收益
        n_days = len(equity)
        n_years = n_days / 252
        annualized_return = (final_value / self.initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 最大回撤
        cummax = total_value.cummax()
        drawdown = (total_value - cummax) / cummax
        max_drawdown = drawdown.min()

        # 夏普比率
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 胜率
        if self.trades:
            sell_trades = [t for t in self.trades if t.direction == 'SELL']
            if sell_trades:
                wins = sum(1 for t in sell_trades if t.price > self.initial_capital)
                win_rate = wins / len(sell_trades)
                gross_profit = sum(t.price * t.quantity for t in sell_trades if t.price > 0)
                gross_loss = sum(t.price * t.quantity for t in sell_trades if t.price <= 0)
                profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0

        return BacktestReport(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            equity_curve=equity,
            trades=self.trades
        )
