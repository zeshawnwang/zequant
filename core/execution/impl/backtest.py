"""回测引擎

支持两种策略类型的统一回测引擎：
1. QuantStrategy：旧架构（Selector + Timing + Portfolio）
2. SignalStrategy：新架构（Selector + PositionSizer + Composer + RiskManager）

设计要点：
- 事件驱动，每日按序执行：止损检查 → 策略下单 → 净值计算
- T+1约束：当日买入的股票不能当日卖出
- Universe过滤：自动剔除ST/新股/涨跌停/停牌
- 订单延迟到次日开盘执行
- FIFO配对计算盈亏统计
"""
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...risk.fee import FeeCalculator
from ...risk import RiskManager
from ...screening.universe import SymbolUniverse
from ...strategies.base.strategy import SignalStrategy, TargetPosition

# 兼容旧架构的 Order 和 Position
try:
    from ...strategies.base.strategy import Order, Position
except ImportError:
    @dataclass
    class Order:
        symbol: str
        direction: str
        quantity: int
        price: float = 0.0
        reason: str = ""
    
    @dataclass
    class Position:
        symbol: str
        quantity: int
        entry_price: float
        entry_date: str = ""

logger = logging.getLogger(__name__)


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
class FinalPosition:
    """回测结束时的持仓快照"""
    symbol: str
    quantity: int
    entry_price: float
    last_price: float
    market_value: float
    pnl: float
    pnl_pct: float


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
    initial_capital: float = 0.0
    final_value: float = 0.0
    final_cash: float = 0.0
    final_position_value: float = 0.0
    final_positions: List[FinalPosition] = field(default_factory=list)
    selection_log: List[dict] = field(default_factory=list)
    strategy_name: str = ""
    selector_description: str = ""
    factors_used: Dict[str, float] = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""

    def pretty_print(self, top_positions: int = 20, top_selections: int = 5) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("回测报告")
        lines.append("=" * 72)
        if self.strategy_name:
            lines.append(f"策略名称   : {self.strategy_name}")
        if self.selector_description:
            lines.append(f"选股逻辑   : {self.selector_description}")
        if self.start_date or self.end_date:
            lines.append(f"回测区间   : {self.start_date} ~ {self.end_date}")
        lines.append("")
        lines.append("─── 资金 ───")
        lines.append(f"初始本金   : {self.initial_capital:>15,.2f}")
        lines.append(f"期末现金   : {self.final_cash:>15,.2f}")
        lines.append(f"期末持仓   : {self.final_position_value:>15,.2f}")
        lines.append(f"期末总值   : {self.final_value:>15,.2f}")
        lines.append(f"绝对盈亏   : {self.final_value - self.initial_capital:>+15,.2f}")
        lines.append("")
        lines.append("─── 收益指标 ───")
        lines.append(f"总收益率   : {self.total_return*100:>+8.2f}%")
        lines.append(f"年化收益   : {self.annualized_return*100:>+8.2f}%")
        lines.append(f"最大回撤   : {self.max_drawdown*100:>+8.2f}%")
        lines.append(f"夏普比率   : {self.sharpe_ratio:>+8.2f}")
        lines.append(f"胜率       : {self.win_rate*100:>+8.2f}%")
        lines.append(f"盈亏比     : {self.profit_factor:>8.2f}")
        lines.append(f"交易次数   : {self.total_trades:>8d}")
        lines.append("")
        if self.factors_used:
            lines.append("─── 使用的因子(权重) ───")
            for f, w in sorted(self.factors_used.items(), key=lambda x: -abs(x[1])):
                lines.append(f"  {f:<25s} {w:>+.4f}")
            lines.append("")
        if self.final_positions:
            lines.append(f"─── 末日持仓 (共 {len(self.final_positions)} 只,显示前 {top_positions}) ───")
            lines.append(f"  {'symbol':<10} {'qty':>8} {'entry':>10} {'last':>10} {'mv':>14} {'pnl':>12} {'pnl%':>8}")
            for p in sorted(self.final_positions, key=lambda x: -x.market_value)[:top_positions]:
                lines.append(
                    f"  {p.symbol:<10} {p.quantity:>8d} {p.entry_price:>10.3f} "
                    f"{p.last_price:>10.3f} {p.market_value:>14,.2f} "
                    f"{p.pnl:>+12,.2f} {p.pnl_pct*100:>+7.2f}%"
                )
            lines.append("")
        if self.selection_log:
            lines.append(f"─── 选股记录 (共 {len(self.selection_log)} 个调仓日,显示首/末 {top_selections}) ───")
            head = self.selection_log[:top_selections]
            tail = self.selection_log[-top_selections:] if len(self.selection_log) > top_selections else []
            for rec in head:
                syms = rec.get("selected", [])
                preview = ", ".join(syms[:10]) + (" ..." if len(syms) > 10 else "")
                lines.append(f"  {rec['date']}  ({len(syms)}) {preview}")
            if tail and tail != head:
                lines.append(f"  ... 中间 {len(self.selection_log) - 2*top_selections} 天 ...")
                for rec in tail:
                    syms = rec.get("selected", [])
                    preview = ", ".join(syms[:10]) + (" ..." if len(syms) > 10 else "")
                    lines.append(f"  {rec['date']}  ({len(syms)}) {preview}")
        lines.append("=" * 72)
        return "\n".join(lines)


class BacktestEngine:
    """统一回测引擎

    支持两种策略类型：
    - QuantStrategy：旧架构
    - SignalStrategy：新架构（信号流驱动）
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        fee_config: dict = None,
        risk_config: dict = None,
        universe: Optional[SymbolUniverse] = None,
    ):
        self.initial_capital = initial_capital
        self.fee_calc = FeeCalculator(fee_config or {})
        rcfg = dict(risk_config or {})
        if fee_config and "fees" not in rcfg:
            rcfg["fees"] = fee_config
        self.risk_mgr = RiskManager(rcfg)
        self.universe = universe
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.trades: List[Trade] = []
        self.daily_values: List[dict] = []
        self._buy_date_map: Dict[str, str] = {}
        self.selection_log: List[dict] = []
        self._last_price_map: Dict[str, float] = {}
        self._strategy_ref: Optional[SignalStrategy] = None
        self._start_date: str = ""
        self._end_date: str = ""
        self._pending_orders: List[Order] = []

    def run(
        self,
        strategy: SignalStrategy,
        factor_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        rebalance_freq: str = '1d',
    ) -> BacktestReport:
        """执行回测

        Args:
            strategy: 策略实例（QuantStrategy或SignalStrategy）
            factor_data: 因子数据，包含date、symbol、及各因子列
            start_date: 回测开始日期
            end_date: 回测结束日期
            rebalance_freq: 调仓频率（暂未实现多频率）
        """
        if 'date' not in factor_data.columns:
            raise ValueError("factor_data must have 'date' column")

        df = factor_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[
            (df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date))
        ]
        if df.empty:
            logger.warning("回测区间内无数据")
            return self._generate_report()

        dates = sorted(df['date'].unique())
        self._strategy_ref = strategy
        self._start_date = str(pd.Timestamp(dates[0]).date())
        self._end_date = str(pd.Timestamp(dates[-1]).date())

        for current_date in dates:
            day_slice = df[df['date'] <= current_date]
            today_only = df[df['date'] == current_date]
            date_str = str(pd.Timestamp(current_date).date())

            self._execute_pending_orders(today_only, date_str)

            buyable_today = self._get_buyable_symbols(current_date, today_only)

            stop_orders = self._check_stops(today_only, date_str, buyable_today)
            self._pending_orders.extend(stop_orders)

            orders = self._generate_orders_signal(strategy, day_slice, date_str)

            sel = list(getattr(strategy, "last_selected", []) or [])
            if sel:
                self.selection_log.append({
                    "date": date_str,
                    "selected": sel,
                    "n": len(sel),
                })

            for order in orders:
                order = self._validate_order(order, date_str, buyable_today)
                if order:
                    self._pending_orders.append(order)

            self._update_daily_values(today_only)

            if today_only is not None and not today_only.empty:
                for sym, px in zip(today_only['symbol'], today_only['close']):
                    if px is not None and not pd.isna(px):
                        self._last_price_map[sym] = float(px)

        return self._generate_report()

    def _execute_pending_orders(self, today_only: pd.DataFrame, date_str: str):
        """执行前一日pending订单（用今日开盘价）"""
        if not self._pending_orders:
            return

        open_map = {}
        if today_only is not None and not today_only.empty:
            open_map = dict(zip(today_only['symbol'], today_only['open']))

        executed = []
        for order in self._pending_orders:
            exec_price = open_map.get(order.symbol)
            if exec_price is not None and not pd.isna(exec_price) and exec_price > 0:
                order.price = float(exec_price)
                self._execute_order(order, date_str)
                executed.append(order)

        self._pending_orders = [o for o in self._pending_orders if o not in executed]

    def _get_buyable_symbols(self, current_date, today_only):
        """获取当日可买入的股票列表"""
        if self.universe is None:
            return set(today_only['symbol'].tolist()) if today_only is not None and not today_only.empty else set()

        buyable = self.universe.filter_buyable(current_date, today_only)
        visible = buyable | set(self.positions.keys())
        return visible

    def _generate_orders_signal(
        self,
        strategy: SignalStrategy,
        day_slice: pd.DataFrame,
        date_str: str,
    ) -> List[Order]:
        """为SignalStrategy生成订单"""
        try:
            return strategy.generate_orders(date_str, day_slice, self.cash, self.positions)
        except Exception as e:
            logger.error(f"SignalStrategy生成订单失败: {e}")
            return []



    def _validate_order(
        self,
        order: Order,
        date_str: str,
        buyable_today: set,
    ) -> Optional[Order]:
        """验证订单有效性"""
        if order.direction == 'BUY':
            if buyable_today and order.symbol not in buyable_today:
                return None
        elif order.direction == 'SELL':
            if self._buy_date_map.get(order.symbol) == date_str:
                return None
            if self.universe is not None:
                return None
        return order

    def _update_daily_values(self, today_only: pd.DataFrame):
        """更新每日净值记录"""
        portfolio_value = self._calc_portfolio_value(today_only)
        self.daily_values.append({
            'date': today_only['date'].iloc[0] if today_only is not None and not today_only.empty else None,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'total_value': self.cash + portfolio_value,
        })

    def _execute_order(self, order: Order, date: str):
        """执行订单"""
        if order.quantity <= 0 or order.price <= 0:
            return

        exec_price = self.risk_mgr.apply_slippage(order.price, order.direction.lower())
        if exec_price <= 0:
            return

        if order.direction == 'BUY':
            self._execute_buy(order, exec_price, date)
        elif order.direction == 'SELL':
            self._execute_sell(order, exec_price, date)

    def _execute_buy(self, order: Order, exec_price: float, date: str):
        """执行买入"""
        cost = self.fee_calc.calc_net_proceed('buy', order.symbol, exec_price, order.quantity)
        qty = order.quantity

        if cost > self.cash:
            max_qty = int(self.cash / exec_price / 100) * 100
            if max_qty < 100:
                return
            qty = max_qty
            cost = self.fee_calc.calc_net_proceed('buy', order.symbol, exec_price, qty)
            if cost > self.cash:
                return

        self.cash -= cost

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

        self._buy_date_map[order.symbol] = date
        self.trades.append(Trade(
            date=date,
            symbol=order.symbol,
            direction='BUY',
            price=exec_price,
            quantity=qty,
            commission=self.fee_calc.calc_buy(order.symbol, exec_price, qty).total,
        ))

    def _execute_sell(self, order: Order, exec_price: float, date: str):
        """执行卖出"""
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
            date=date,
            symbol=order.symbol,
            direction='SELL',
            price=exec_price,
            quantity=sell_qty,
            commission=self.fee_calc.calc_sell(order.symbol, exec_price, sell_qty).total,
        ))

    def _check_stops(
        self,
        today_only: pd.DataFrame,
        date_str: str,
        buyable_today: set,
    ) -> List[Order]:
        """止损止盈检查"""
        orders: List[Order] = []
        if today_only is None or today_only.empty or not self.positions:
            return orders

        price_map = dict(zip(today_only['symbol'], today_only['close']))
        to_sell = []

        for symbol, pos in self.positions.items():
            if self._buy_date_map.get(symbol) == date_str:
                continue

            cp = price_map.get(symbol)
            if cp is None or pd.isna(cp):
                continue
            cp = float(cp)

            should_stop, reason = self.risk_mgr.check_stop_loss(
                pos.entry_price, cp, direction='long'
            )
            if not should_stop:
                continue

            if self.universe is not None and not self.universe.is_sellable(symbol, today_only):
                continue

            to_sell.append((symbol, cp, reason))

        for symbol, cp, reason in to_sell:
            orders.append(Order(
                symbol=symbol,
                direction='SELL',
                quantity=self.positions[symbol].quantity,
                price=cp,
                reason=reason,
            ))

        return orders

    def _calc_portfolio_value(self, today_only: pd.DataFrame) -> float:
        """计算组合市值"""
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
        """生成回测报告"""
        equity = pd.DataFrame(self.daily_values)

        if equity.empty:
            return BacktestReport(
                0, 0, 0, 0, 0, 0, 0, equity,
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                final_cash=self.cash,
                start_date=self._start_date,
                end_date=self._end_date,
                selection_log=self.selection_log,
            )

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

        win_rate, profit_factor = self._calc_pnl_stats()

        final_positions: List[FinalPosition] = []
        final_position_value = 0.0
        for sym, pos in self.positions.items():
            last_px = self._last_price_map.get(sym, pos.entry_price)
            mv = pos.quantity * float(last_px)
            pnl = (float(last_px) - pos.entry_price) * pos.quantity
            pnl_pct = (float(last_px) - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            final_positions.append(FinalPosition(
                symbol=sym,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                last_price=float(last_px),
                market_value=mv,
                pnl=pnl,
                pnl_pct=pnl_pct,
            ))
            final_position_value += mv

        strategy_name, selector_desc, factors_used = self._extract_strategy_info()

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
            initial_capital=float(self.initial_capital),
            final_value=float(final_value),
            final_cash=float(self.cash),
            final_position_value=float(final_position_value),
            final_positions=final_positions,
            selection_log=self.selection_log,
            strategy_name=strategy_name,
            selector_description=selector_desc,
            factors_used=factors_used,
            start_date=self._start_date,
            end_date=self._end_date,
        )

    def _extract_strategy_info(self):
        """提取策略信息"""
        strategy_name = ""
        selector_desc = ""
        factors_used: Dict[str, float] = {}

        if self._strategy_ref is None:
            return strategy_name, selector_desc, factors_used

        strategy_name = self._strategy_ref.name

        if isinstance(self._strategy_ref, SignalStrategy):
            selector = self._strategy_ref.selector
            if hasattr(selector, 'get_description'):
                try:
                    selector_desc = selector.get_description()
                except Exception:
                    selector_desc = selector.__class__.__name__
            else:
                selector_desc = selector.__class__.__name__

            if hasattr(selector, 'factors') and isinstance(selector.factors, list):
                factors_used = {f: 1.0 for f in selector.factors}
            elif hasattr(selector, 'weights') and isinstance(selector.weights, dict):
                factors_used = dict(selector.weights)
        else:
            selector = self._strategy_ref.selector
            if hasattr(selector, 'get_description'):
                try:
                    selector_desc = selector.get_description()
                except Exception:
                    selector_desc = selector.__class__.__name__
            else:
                selector_desc = selector.__class__.__name__

            if hasattr(selector, 'weights') and isinstance(selector.weights, dict):
                factors_used = dict(selector.weights)
            elif hasattr(selector, 'factor_name'):
                factors_used = {getattr(selector, 'factor_name'): 1.0}

        return strategy_name, selector_desc, factors_used

    def _calc_pnl_stats(self):
        """FIFO配对计算盈亏统计"""
        buys: Dict[str, deque] = defaultdict(deque)
        realized = []

        for t in self.trades:
            if t.direction == 'BUY':
                buys[t.symbol].append([t.price, t.quantity])
            else:
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
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0
            else (999.0 if gross_profit > 0 else 0.0)
        )
        return float(win_rate), float(profit_factor)
