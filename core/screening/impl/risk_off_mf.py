"""RiskOffMultiFactorSelector — V7.1 多因子 + 信号层风险熔断。

继承 MultiFactorSelector, 在 score 计算后应用 risk_off 减权:
  - 单指标模式 (m5_only): momentum_5 z-score < risk_off_z → score × risk_off_scale
  - 双指标模式 (m5_and_vol / m5_or_vol / low_m5_low_vol): 加入 volatility_20 联动

返回 {symbol: score} 字典而非排序列表, 供 SignalStrategy 后续 RP 仓位分配使用。

参数
----
weights : dict[str, float]
    多因子权重, 与 MultiFactorSelector 一致
risk_off_z : float
    m5 触发 z 阈值 (默认 -2.5)
risk_off_scale : float
    触发后 score 乘数 (默认 0.55)
risk_off_mode : str
    "m5_only" / "m5_and_vol" / "m5_or_vol" / "low_m5_low_vol"
risk_off_vol_z : float
    vol 触发阈值 (双指标模式用)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd

from .multi_factor import MultiFactorSelector, _zscore

logger = logging.getLogger(__name__)


class RiskOffMultiFactorSelector(MultiFactorSelector):
    """V7.1 多因子 + 信号层风险熔断 selector。"""

    def __init__(
        self,
        weights: Dict[str, float],
        top_n: int = 100,
        winsorize: float = 0.01,
        normalize_weights: bool = True,
        min_factors_coverage: int = 1,
        risk_off_z: float = -2.5,
        risk_off_scale: float = 0.55,
        risk_off_mode: str = "m5_only",
        risk_off_vol_z: float = 1.5,
        trigger_factor: str = "momentum_5",
    ):
        super().__init__(
            weights=weights,
            top_n=top_n,
            winsorize=winsorize,
            normalize_weights=normalize_weights,
            min_factors_coverage=min_factors_coverage,
        )
        self.risk_off_z = float(risk_off_z)
        self.risk_off_scale = float(risk_off_scale)
        self.risk_off_mode = str(risk_off_mode)
        self.risk_off_vol_z = float(risk_off_vol_z)
        self.trigger_factor = str(trigger_factor)

    def _apply_risk_off(self, latest: pd.DataFrame, score: pd.Series) -> pd.Series:
        """应用 risk_off 减权到 score series."""
        if self.trigger_factor not in latest.columns:
            logger.warning(f"  risk_off: 触发因子 {self.trigger_factor} 不存在, 跳过熔断")
            return score

        trigger_z = _zscore(latest[self.trigger_factor], winsorize=self.winsorize)
        trigger_arr = trigger_z.values

        if self.risk_off_mode == "m5_only":
            mask = trigger_arr < self.risk_off_z
        elif self.risk_off_mode == "m5_and_vol" and "volatility_20" in latest.columns:
            vol = _zscore(latest["volatility_20"], winsorize=self.winsorize).values
            mask = (trigger_arr < self.risk_off_z) & (vol > self.risk_off_vol_z)
        elif self.risk_off_mode == "m5_or_vol" and "volatility_20" in latest.columns:
            vol = _zscore(latest["volatility_20"], winsorize=self.winsorize).values
            mask = (trigger_arr < self.risk_off_z) | (vol > self.risk_off_vol_z)
        elif self.risk_off_mode == "low_m5_low_vol" and "volatility_20" in latest.columns:
            vol = _zscore(latest["volatility_20"], winsorize=self.winsorize).values
            mask = (trigger_arr < self.risk_off_z) & (vol < self.risk_off_vol_z)
        else:
            mask = trigger_arr < self.risk_off_z

        n = int(mask.sum())
        if n > 0:
            new_vals = score.values.astype(float).copy()
            new_vals[mask] *= self.risk_off_scale
            score = pd.Series(new_vals, index=score.index)
            logger.info(
                f"  risk_off: mode={self.risk_off_mode} "
                f"{self.trigger_factor}<{self.risk_off_z} x{self.risk_off_scale}: "
                f"{n} 票减权"
            )
        return score

    def get_scores(
        self,
        factor_data: pd.DataFrame,
        date,
    ) -> Dict[str, float]:
        """返回 {symbol: score} 字典 (供 SignalStrategy RP 仓位使用).

        Args:
            factor_data: 历史因子面板 (含 date / symbol / 因子列)
            date: 当前日期
        """
        if factor_data is None or factor_data.empty:
            return {}
        df = factor_data
        if "date" in df.columns:
            df = df[df["date"] <= date]
        if df.empty:
            return {}

        latest = df.sort_values("date").groupby("symbol").tail(1)
        if latest.empty:
            return {}
        latest = latest.set_index("symbol")

        score = self._compute_score(latest)
        if score.empty:
            return {}

        score = self._apply_risk_off(latest, score)
        return score.to_dict()

    def select(
        self,
        factor_data: pd.DataFrame,
        date,
        top_n: int,
    ) -> List[str]:
        """重写 select, 走 get_scores 路径以应用 risk_off."""
        scores = self.get_scores(factor_data, date)
        if not scores:
            return []
        n = top_n or self.top_n
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:n]
        return [s for s, _ in ranked]

    def get_description(self) -> str:
        base = super().get_description()
        return (
            f"RiskOffMF[{self.risk_off_mode}, "
            f"z<{self.risk_off_z}*x{self.risk_off_scale}]"
            f"({base})"
        )

    @classmethod
    def compute_signal(
        cls,
        latest: pd.DataFrame,
        weights: Dict[str, float],
        top_n: int = 10,
        winsorize: float = 0.01,
        risk_off_z: float = -2.5,
        risk_off_scale: float = 0.55,
        trigger_factor: str = "momentum_5",
    ) -> "SignalResult":
        """单信号截面计算（供 live/回测/模拟器统一调用）。

        与 evaluate.py 的 build_v71_mf_signal 对齐：
            截面 zscore → 加权 → risk_off 减权 → 取 top_n
        """
        from dataclasses import dataclass

        @dataclass
        class SignalResult:
            score: pd.Series
            risk_off_triggered: list
            latest: pd.DataFrame

        if latest is None or latest.empty:
            return SignalResult(pd.Series(dtype=float), [], latest)

        sel = cls(
            weights=weights,
            top_n=top_n,
            winsorize=winsorize,
            risk_off_z=risk_off_z,
            risk_off_scale=risk_off_scale,
            trigger_factor=trigger_factor,
        )
        latest_idx = latest.set_index("symbol") if "symbol" in latest.columns else latest
        score = sel._compute_score(latest_idx)
        if score.empty:
            return SignalResult(pd.Series(dtype=float), [], latest)

        score = sel._apply_risk_off(latest_idx, score)

        risk_off_triggered = []
        if trigger_factor in latest_idx.columns:
            z = _zscore(latest_idx[trigger_factor], winsorize=winsorize)
            risk_off_triggered = [s for s in score.index if s in z.index and z[s] < risk_off_z]

        return SignalResult(score, risk_off_triggered, latest)