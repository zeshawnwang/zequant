"""因子衰减监控模块。

滚动窗口计算因子 IC/IR 时序，检测因子有效性是否在衰减。

功能：
- 滚动窗口 IC 时序计算
- IR 衰减检测与预警
- 因子健康度评分
- 实盘因子权重联动建议

用法：
    from core.research.impl.factor_monitor import FactorDecayMonitor

    monitor = FactorDecayMonitor(db)
    report = monitor.run(factor_names=[...], end_date="2026-06-10")
    alerts = report.get_alerts()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import FactorEvaluator, FactorEvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class FactorHealthRecord:
    """单因子单窗口的健康记录。"""
    factor_name: str
    window_end: str
    ic_mean: float
    ir: float
    ic_t_stat: float
    top_group_return: float


@dataclass
class FactorAlert:
    """因子衰减预警。"""
    factor_name: str
    alert_type: str  # "decay" | "reversal" | "dead"
    severity: str    # "warning" | "critical"
    message: str
    current_ir: float
    historical_ir: float
    decay_ratio: float  # current / historical


@dataclass
class MonitorReport:
    """监控报告。"""
    run_date: str
    end_date: str
    n_factors: int
    window_days: int
    step_days: int
    n_windows: int
    records: pd.DataFrame  # 全量 IC 时序记录
    alerts: List[FactorAlert] = field(default_factory=list)
    health_scores: Dict[str, float] = field(default_factory=dict)

    def get_alerts(self, severity: str = None) -> List[FactorAlert]:
        """获取预警列表，可按严重度过滤。"""
        if severity:
            return [a for a in self.alerts if a.severity == severity]
        return self.alerts

    def summary(self) -> str:
        """输出摘要。"""
        lines = [
            f"=== 因子衰减监控报告 ({self.run_date}) ===",
            f"监控截止: {self.end_date} | 因子数: {self.n_factors} | 窗口: {self.window_days}天 × {self.n_windows}期",
            f"预警: {len(self.alerts)} 条 (critical: {len(self.get_alerts('critical'))}, warning: {len(self.get_alerts('warning'))})",
            "",
        ]
        if self.alerts:
            lines.append("--- 预警详情 ---")
            for a in sorted(self.alerts, key=lambda x: x.decay_ratio):
                icon = "🔴" if a.severity == "critical" else "🟡"
                lines.append(f"  {icon} {a.factor_name}: {a.message}")
            lines.append("")

        # 健康度排名
        if self.health_scores:
            lines.append("--- 因子健康度 (0~1, 越高越好) ---")
            sorted_scores = sorted(self.health_scores.items(), key=lambda x: x[1])
            for name, score in sorted_scores[:10]:
                status = "⚠️" if score < 0.5 else "✅"
                lines.append(f"  {status} {name}: {score:.3f}")
            if len(sorted_scores) > 10:
                lines.append(f"  ... 共 {len(sorted_scores)} 个因子")

        return "\n".join(lines)


class FactorDecayMonitor:
    """因子衰减监控器。

    通过滚动窗口计算 IC/IR 时序，检测因子是否正在失效。

    参数：
        db: Database 实例
        window_days: 滚动窗口大小（交易日）
        step_days: 滑动步长（交易日）
        decay_threshold: IR 衰减到历史均值的比例阈值（低于此发 warning）
        dead_threshold: IR 衰减到历史均值的比例阈值（低于此发 critical）
        min_windows: 最少需要多少个窗口才出预警
    """

    def __init__(
        self,
        db,
        window_days: int = 120,
        step_days: int = 20,
        decay_threshold: float = 0.5,
        dead_threshold: float = 0.2,
        min_windows: int = 5,
    ):
        self.db = db
        self.evaluator = FactorEvaluator(db)
        self.window_days = window_days
        self.step_days = step_days
        self.decay_threshold = decay_threshold
        self.dead_threshold = dead_threshold
        self.min_windows = min_windows

    def run(
        self,
        factor_names: List[str],
        end_date: str,
        lookback_years: float = 3.0,
        forward_days: int = 5,
    ) -> MonitorReport:
        """执行因子衰减监控。

        Args:
            factor_names: 要监控的因子名列表
            end_date: 监控截止日期
            lookback_years: 往前看多少年
            forward_days: IC 计算的前瞻收益天数

        Returns:
            MonitorReport
        """
        from datetime import datetime, timedelta

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - pd.DateOffset(years=lookback_years)
        start_date = start_dt.strftime("%Y-%m-%d")

        # 获取交易日序列
        trading_dates = self._get_trading_dates(start_date, end_date)
        if len(trading_dates) < self.window_days + self.step_days:
            logger.warning(f"交易日不足: {len(trading_dates)} < {self.window_days + self.step_days}")
            return MonitorReport(
                run_date=str(datetime.now().date()),
                end_date=end_date,
                n_factors=len(factor_names),
                window_days=self.window_days,
                step_days=self.step_days,
                n_windows=0,
                records=pd.DataFrame(),
            )

        # 构建滚动窗口
        windows = []
        i = self.window_days
        while i <= len(trading_dates):
            w_start = trading_dates[i - self.window_days]
            w_end = trading_dates[i - 1]
            windows.append((w_start, w_end))
            i += self.step_days

        logger.info(f"因子监控: {len(factor_names)} 因子 × {len(windows)} 窗口")

        # 逐窗口计算
        records = []
        for w_start, w_end in windows:
            for factor_name in factor_names:
                try:
                    result = self.evaluator.evaluate(
                        factor_name, (w_start, w_end), forward_days
                    )
                    records.append(FactorHealthRecord(
                        factor_name=factor_name,
                        window_end=w_end,
                        ic_mean=result.ic_mean,
                        ir=result.ir,
                        ic_t_stat=result.ic_t_stat,
                        top_group_return=result.top_group_return,
                    ))
                except Exception as e:
                    logger.debug(f"  {factor_name} @ {w_end}: {e}")

        if not records:
            return MonitorReport(
                run_date=str(datetime.now().date()),
                end_date=end_date,
                n_factors=len(factor_names),
                window_days=self.window_days,
                step_days=self.step_days,
                n_windows=len(windows),
                records=pd.DataFrame(),
            )

        df = pd.DataFrame([
            {"factor": r.factor_name, "date": r.window_end,
             "ic_mean": r.ic_mean, "ir": r.ir,
             "ic_t_stat": r.ic_t_stat, "top_ret": r.top_group_return}
            for r in records
        ])

        # 生成预警和健康度
        alerts = self._detect_decay(df)
        health_scores = self._calc_health_scores(df)

        from datetime import datetime as dt
        return MonitorReport(
            run_date=str(dt.now().date()),
            end_date=end_date,
            n_factors=len(factor_names),
            window_days=self.window_days,
            step_days=self.step_days,
            n_windows=len(windows),
            records=df,
            alerts=alerts,
            health_scores=health_scores,
        )

    def _detect_decay(self, df: pd.DataFrame) -> List[FactorAlert]:
        """检测因子衰减。"""
        alerts = []
        for factor_name in df["factor"].unique():
            sub = df[df["factor"] == factor_name].sort_values("date")
            if len(sub) < self.min_windows:
                continue

            hist_ir = sub["ir"].mean()
            hist_ir_abs = abs(hist_ir)
            if hist_ir_abs < 0.01:
                continue  # 历史 IR 就很低，不监控

            # 近3期 IR 均值
            recent = sub.tail(3)
            recent_ir = recent["ir"].mean()
            recent_ir_abs = abs(recent_ir)

            # 方向反转检测
            if hist_ir > 0.02 and recent_ir < -0.01:
                alerts.append(FactorAlert(
                    factor_name=factor_name,
                    alert_type="reversal",
                    severity="critical",
                    message=f"IC方向反转! 历史IR={hist_ir:.3f} → 近期IR={recent_ir:.3f}",
                    current_ir=recent_ir,
                    historical_ir=hist_ir,
                    decay_ratio=recent_ir / hist_ir if hist_ir != 0 else 0,
                ))
                continue

            # 衰减检测
            decay_ratio = recent_ir_abs / hist_ir_abs
            if decay_ratio < self.dead_threshold:
                alerts.append(FactorAlert(
                    factor_name=factor_name,
                    alert_type="dead",
                    severity="critical",
                    message=f"因子近乎失效: IR从{hist_ir:.3f}衰减到{recent_ir:.3f} (衰减{(1-decay_ratio)*100:.0f}%)",
                    current_ir=recent_ir,
                    historical_ir=hist_ir,
                    decay_ratio=decay_ratio,
                ))
            elif decay_ratio < self.decay_threshold:
                alerts.append(FactorAlert(
                    factor_name=factor_name,
                    alert_type="decay",
                    severity="warning",
                    message=f"因子衰减: IR从{hist_ir:.3f}降到{recent_ir:.3f} (衰减{(1-decay_ratio)*100:.0f}%)",
                    current_ir=recent_ir,
                    historical_ir=hist_ir,
                    decay_ratio=decay_ratio,
                ))

        return alerts

    def _calc_health_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算因子健康度评分 (0~1)。

        综合考虑：
        - IR 绝对值稳定性 (40%)
        - 近期 vs 历史 IR 比值 (40%)
        - IC 方向一致性 (20%)
        """
        scores = {}
        for factor_name in df["factor"].unique():
            sub = df[df["factor"] == factor_name].sort_values("date")
            if len(sub) < self.min_windows:
                scores[factor_name] = 0.5  # 数据不足，给中性分
                continue

            ir_series = sub["ir"]
            hist_ir_abs = abs(ir_series.mean())
            recent_ir_abs = abs(sub.tail(3)["ir"].mean())

            # 1. IR 稳定性: std / mean 越小越好
            stability = 1.0 - min(ir_series.std() / (hist_ir_abs + 1e-6), 2.0) / 2.0
            stability = max(0, stability)

            # 2. 近期/历史比值
            ratio = min(recent_ir_abs / (hist_ir_abs + 1e-6), 1.5) / 1.5

            # 3. IC 方向一致性: 正IC占比
            ic_series = sub["ic_mean"]
            if hist_ir_abs > 0.01:
                direction = ir_series.mean()
                if direction > 0:
                    consistency = (ic_series > 0).mean()
                else:
                    consistency = (ic_series < 0).mean()
            else:
                consistency = 0.5

            score = stability * 0.4 + ratio * 0.4 + consistency * 0.2
            scores[factor_name] = round(score, 4)

        return scores

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """从数据库获取交易日序列。"""
        rows = self.db.conn.execute(
            "SELECT DISTINCT date FROM daily_bars WHERE date >= ? AND date <= ? ORDER BY date",
            [start_date, end_date]
        ).fetchall()
        return [str(r[0]) for r in rows]