"""实盘监控 — 日报/周报/警报。"""
from __future__ import annotations
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Dashboard:
    @staticmethod
    def generate_report() -> str:
        """生成今日绩效报告文件。"""
        today = datetime.now().strftime("%Y-%m-%d")
        # TODO: 从 live_data.db 读取实际数据计算
        logger.info("日报生成: %s (MVP占位)", today)
        return f"日报 {today}: 待接入持仓数据"

    @staticmethod
    def check_alerts(position_value: float, cash: float, max_dd: float) -> list:
        """检查预警条件。"""
        alerts = []
        if max_dd < -0.20:
            alerts.append(f"🔴 最大回撤超过20%: {max_dd:.1%}")
        if position_value == 0 and cash > 0:
            alerts.append("🟡 空仓")
        return alerts
