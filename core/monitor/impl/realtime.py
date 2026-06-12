"""实时监控模块

功能：
- 实时监控信号、订单、持仓状态
- 阈值告警（回撤、亏损、换手等）
- 状态快照和历史记录
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警信息"""
    timestamp: datetime
    level: AlertLevel
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class MonitorConfig:
    """监控配置"""
    max_drawdown_threshold: float = 0.15
    max_daily_loss: float = 0.05
    max_position_concentration: float = 0.25
    alert_on_trade: bool = True
    alert_on_rebalance: bool = True
    alert_callback: Optional[Callable] = None


@dataclass
class PortfolioSnapshot:
    """组合快照"""
    timestamp: datetime
    cash: float
    portfolio_value: float
    total_value: float
    positions: Dict[str, int]
    pending_orders: List[dict]
    daily_return: float = 0.0
    cumulative_return: float = 0.0


class RealtimeMonitor:
    """实时监控器

    监控实盘/模拟交易的运行状态
    """

    def __init__(self, config: MonitorConfig = None):
        self.config = config or MonitorConfig()
        self._snapshots: List[PortfolioSnapshot] = []
        self._alerts: List[Alert] = []
        self._trades_today: List[dict] = []
        self._peak_value: float = 0.0
        self._last_value: float = 0.0

    def record_trade(self, trade: dict):
        """记录交易"""
        trade['timestamp'] = datetime.now()
        self._trades_today.append(trade)

        if self.config.alert_on_trade:
            qty = trade.get('shares', trade.get('quantity', 0))
            msg = f"交易: {trade['direction']} {trade['symbol']} {qty}股 @ {trade['price']}"
            self._add_alert(AlertLevel.INFO, msg, trade)

    def record_snapshot(self, snapshot: PortfolioSnapshot):
        """记录快照"""
        self._snapshots.append(snapshot)

        if self._peak_value == 0:
            self._peak_value = snapshot.total_value

        if snapshot.total_value > self._peak_value:
            self._peak_value = snapshot.total_value

        if self._last_value > 0:
            snapshot.daily_return = (snapshot.total_value - self._last_value) / self._last_value
        cumulative = (snapshot.total_value - self._snapshots[0].total_value) / self._snapshots[0].total_value if len(self._snapshots) > 1 and self._snapshots[0].total_value > 0 else 0
        snapshot.cumulative_return = cumulative

        self._check_alerts(snapshot)
        self._last_value = snapshot.total_value

    def _check_alerts(self, snapshot: PortfolioSnapshot):
        """检查告警条件"""
        if self.config.max_drawdown_threshold > 0:
            drawdown = (self._peak_value - snapshot.total_value) / self._peak_value if self._peak_value > 0 else 0
            if drawdown > self.config.max_drawdown_threshold:
                self._add_alert(
                    AlertLevel.CRITICAL,
                    f"最大回撤超限: {drawdown*100:.2f}% > {self.config.max_drawdown_threshold*100:.2f}%",
                    {'drawdown': drawdown, 'threshold': self.config.max_drawdown_threshold}
                )

        if self.config.max_daily_loss > 0:
            if snapshot.daily_return < -self.config.max_daily_loss:
                self._add_alert(
                    AlertLevel.ERROR,
                    f"单日亏损超限: {snapshot.daily_return*100:.2f}% < {-self.config.max_daily_loss*100:.2f}%",
                    {'daily_return': snapshot.daily_return, 'threshold': self.config.max_daily_loss}
                )

        if self.config.max_position_concentration > 0:
            if snapshot.positions:
                max_position_value = max(snapshot.positions.values()) * snapshot.total_value if isinstance(list(snapshot.positions.values())[0], float) else 0
                if max_position_value > self.config.max_position_concentration * snapshot.total_value:
                    max_sym = max(snapshot.positions.items(), key=lambda x: x[1] if isinstance(x[1], float) else 0)[0]
                    self._add_alert(
                        AlertLevel.WARNING,
                        f"仓位集中度过高: {max_sym}",
                        {'symbol': max_sym, 'value': max_position_value}
                    )

    def _add_alert(self, level: AlertLevel, message: str, details: Dict = None):
        """添加告警"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            message=message,
            details=details or {},
        )
        self._alerts.append(alert)
        logger.log(
            logging.WARNING if level in (AlertLevel.WARNING, AlertLevel.ERROR) else logging.INFO,
            f"[{level.value}] {message}"
        )

        if self.config.alert_callback:
            try:
                self.config.alert_callback(alert)
            except Exception as e:
                logger.error(f"告警回调失败: {e}")

    def get_today_trades(self) -> List[dict]:
        """获取今日交易"""
        return list(self._trades_today)

    def get_alerts(self, level: AlertLevel = None, since: datetime = None) -> List[Alert]:
        """获取告警列表"""
        alerts = self._alerts
        if level:
            alerts = [a for a in alerts if a.level == level]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        return alerts

    def get_latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        """获取最新快照"""
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshots(self, since: datetime = None) -> List[PortfolioSnapshot]:
        """获取快照列表"""
        if since is None:
            return list(self._snapshots)
        return [s for s in self._snapshots if s.timestamp >= since]

    def clear_today_data(self):
        """清空今日数据"""
        self._trades_today = []

    def get_summary(self) -> Dict:
        """获取监控摘要"""
        latest = self.get_latest_snapshot()
        return {
            'total_value': latest.total_value if latest else 0,
            'cash': latest.cash if latest else 0,
            'position_value': (latest.total_value - latest.cash) if latest else 0,
            'positions_count': len(latest.positions) if latest else 0,
            'pending_orders': len(latest.pending_orders) if latest else 0,
            'trades_today': len(self._trades_today),
            'alerts_today': len([a for a in self._alerts if a.timestamp.date() == datetime.now().date()]),
            'peak_value': self._peak_value,
            'current_drawdown': (self._peak_value - self._last_value) / self._peak_value if self._peak_value > 0 else 0,
        }
