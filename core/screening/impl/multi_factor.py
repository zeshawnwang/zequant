"""
多因子合成选股器 (MultiFactorSelector)

核心思想:
    score(symbol) = Σ w_i · zscore(factor_i)(截面标准化后加权求和)

关键设计:
    1. 截面 zscore 而非 rank,保留尾部信号强度;支持 winsorize 去极值
    2. 权重可正可负:IR 为负的反转因子权重应为负
    3. 两种构造方式:
       - 手动:MultiFactorSelector({'momentum_20': -1.0, 'volatility_20': -1.0})
       - 自动:MultiFactorSelector.from_registry(db, min_abs_ir=0.2)
                按注册表中 enabled=True 的因子,以各自 IR 作权重
    4. 支持 top_n 构造默认值(与 FactorRankSelector 接口一致)

使用示例:
    sel = MultiFactorSelector({'momentum_20': -0.8, 'volatility_20': -1.2},
                              top_n=100)
    picks = sel.select(factor_data, date='2024-02-01', top_n=50)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..base.selector import IStockSelector


def _zscore(s: pd.Series, winsorize: float = 0.01) -> pd.Series:
    """截面 zscore,先 winsorize 再标准化。winsorize=0.01 表示裁剪上下 1%。"""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s.dropna()
    if s.empty:
        return s
    if 0 < winsorize < 0.5:
        lo, hi = s.quantile(winsorize), s.quantile(1 - winsorize)
        s = s.clip(lo, hi)
    mu, sd = s.mean(), s.std(ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


class MultiFactorSelector(IStockSelector):
    """多因子加权合成选股器。"""

    def __init__(
        self,
        weights: Dict[str, float],
        top_n: int = 100,
        winsorize: float = 0.01,
        normalize_weights: bool = True,
        min_factors_coverage: int = 1,
    ):
        """
        Args:
    weights: {factor_name: weight} 权重可正可负
            top_n: 默认选股数量
            winsorize: 截面去极值分位数,0 表示不裁剪
            normalize_weights: 是否将权重归一化到 Σ|w|=1(便于解释)
            min_factors_coverage: 至少有几个因子非空才计算 score
        """
        if not weights:
            raise ValueError("weights 不能为空")
        self.weights = dict(weights)
        self.top_n = top_n
        self.winsorize = winsorize
        self.min_factors_coverage = max(1, int(min_factors_coverage))
        if normalize_weights:
            s = sum(abs(v) for v in self.weights.values())
            if s > 0:
                self.weights = {k: v / s for k, v in self.weights.items()}

    @property
    def factor_names(self) -> list:
        """该选股器需要消费的因子列(供回测脚本提前加载用)。"""
        return list(self.weights.keys())

    @classmethod
    def from_registry(
        cls,
        db,
        top_n: int = 100,
        min_abs_ir: float = 0.2,
        winsorize: float = 0.01,
    ) -> "MultiFactorSelector":
        """
        从 factor_registry 中拉 enabled=True 且 |IR| >= min_abs_ir 的因子,
        以各自 IR 作为权重(自动识别正负方向)。

        ⚠️ 前视警告:registry 是离线一次性评估的结果,如果评估期与回测期重叠,
            等同于用未来信息当权重。生产化建议用 from_summary(walk-forward 评估)。
        """
        reg = db.get_enabled_factors(min_abs_ir=min_abs_ir, as_dataframe=True)
        if reg is None or reg.empty:
            raise RuntimeError(
                f"factor_registry 中无 |IR| >= {min_abs_ir} 的 enabled 因子,"
                "请先运行 scripts/evaluate_factors.py"
            )
        weights = dict(zip(reg["factor_name"], reg["ir"].astype(float)))
        return cls(weights, top_n=top_n, winsorize=winsorize)

    @classmethod
    def from_summary(
        cls,
        summary,
        top_n: int = 100,
        min_abs_ir: float = 0.2,
        winsorize: float = 0.01,
        weight_col: str = "ir",
    ) -> "MultiFactorSelector":
        """
        从 evaluator.evaluate_all() 返回的 summary DataFrame 直接构造,
        不走数据库,适合 walk-forward 实验:
            ev = FactorEvaluator(db)
            sm = ev.evaluate_all(start_date='2024-01-15', end_date='2024-06-30')
            sel = MultiFactorSelector.from_summary(sm, top_n=60, min_abs_ir=0.2)
            # 然后在 2024-07 ~ 2024-12 做样本外回测

        Args:
            summary: 含 factor_name + ir/ic_mean 列的 DataFrame
            weight_col: 用哪个列作为权重,默认 'ir',也可改 'ic_mean'
        """
        import pandas as pd
        if summary is None or len(summary) == 0:
            raise ValueError("summary 为空")
        if not isinstance(summary, pd.DataFrame):
            summary = pd.DataFrame(summary)
        if "factor_name" not in summary.columns or weight_col not in summary.columns:
            raise ValueError(
                f"summary 必须含 factor_name 和 {weight_col} 列,"
                f"当前列: {list(summary.columns)}"
            )
        df = summary.dropna(subset=[weight_col]).copy()
        df = df[df[weight_col].abs() >= float(min_abs_ir)]
        if df.empty:
            raise RuntimeError(
                f"summary 中无 |{weight_col}| >= {min_abs_ir} 的因子"
            )
        weights = dict(zip(df["factor_name"], df[weight_col].astype(float)))
        return cls(weights, top_n=top_n, winsorize=winsorize)

    def _compute_score(self, latest: pd.DataFrame) -> pd.Series:
        """
        给定某日的截面快照(index=symbol,columns 含各因子列),
        返回 score series。
        """
        used_factors = [f for f in self.weights if f in latest.columns]
        if not used_factors:
            return pd.Series(dtype=float)

        z_df = pd.DataFrame(index=latest.index)
        for f in used_factors:
            z_df[f] = _zscore(latest[f], winsorize=self.winsorize)

        coverage = z_df.notna().sum(axis=1)
        z_df = z_df[coverage >= self.min_factors_coverage]

        if z_df.empty:
            return pd.Series(dtype=float)

        w = pd.Series({f: self.weights[f] for f in used_factors})
        z_df = z_df.fillna(0.0)
        score = z_df.mul(w, axis=1).sum(axis=1)
        return score

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        if factor_data is None or factor_data.empty:
            return []
        df = factor_data
        if "date" in df.columns:
            df = df[df["date"] < date]
        if df.empty:
            return []

        latest = df.sort_values("date").groupby("symbol").tail(1)
        if latest.empty:
            return []
        latest = latest.set_index("symbol")

        score = self._compute_score(latest)
        if score.empty:
            return []

        n = top_n or self.top_n
        ranked = score.sort_values(ascending=False).head(n)
        return ranked.index.tolist()

    def get_description(self) -> str:
        pos = [f"{k}(+{w:.2f})" for k, w in self.weights.items() if w > 0]
        neg = [f"{k}({w:.2f})" for k, w in self.weights.items() if w < 0]
        parts = pos + neg
        return f"MultiFactor[{', '.join(parts)}]"

    def __repr__(self):
        return self.get_description()
