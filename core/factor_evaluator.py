"""因子评估引擎 FactorEvaluator —— 衡量因子的预测力、稳定性、单调性、换手率。

核心指标
--------
- IC (Information Coefficient):因子值与下期收益的截面 Spearman 相关
- IR (Information Ratio):mean(IC) / std(IC),越大越稳定
- IC t-stat:mean(IC) / (std(IC) / sqrt(N)),统计显著性
- 分组收益:按因子分 N 组,看多空收益差(Q_top - Q_bot)
- 单调性:N 组收益是否单调
- 换手率:top 组的股票日度变化率(高换手 = 高交易成本)

实现要点(向量化优化)
----------------------
- 直接消费 [database.py](core/database.py:1) 的因子宽表,避免长表 melt 的 IO/内存开销
- IC 序列用 `groupby(date).corrwith` 一次性向量化计算,不再逐日 apply

用法
----
    from core.factor_evaluator import FactorEvaluator
    ev = FactorEvaluator(db)
    summary = ev.evaluate_all(
        factor_names=["momentum_20", "rsi_14"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        forward_days=5,
    )
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .database import Database

logger = logging.getLogger(__name__)


# ===== 公共字段 ===========================================================
# 评估结果字段(对外稳定契约,registry 表的列与之对应)
EVAL_FIELDS = [
    "factor_name",
    "ic_mean", "ic_std", "ir", "ic_t_stat",
    "turnover", "top_group_ret", "bot_group_ret",
    "monotonic", "n_days",
]


class FactorEvaluator:
    """因子评估引擎(宽表向量化版)。"""

    def __init__(self, db: Database):
        self.db = db

    # ---- 数据准备 --------------------------------------------------------

    def _load_panel(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        forward_days: int,
    ) -> pd.DataFrame:
        """加载评估面板:date / symbol / [因子列...] / fwd_ret。

        - 因子部分直接从宽表读
        - 远期收益 fwd_ret = close_{t+N}/close_t - 1,在 daily_bars 上向后滚动算好,
          并按 (date, symbol) join 回因子表
        """
        wide = self.db.get_factors(
            factor_names=factor_names,
            start_date=start_date,
            end_date=end_date,
            with_close=False,
        )
        if wide is None or wide.empty:
            return pd.DataFrame()
        wide["date"] = pd.to_datetime(wide["date"])

        # 末期需要再向后取 forward_days * 2 + 缓冲,避免 fwd_ret 缺失太多
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=forward_days * 2 + 10)
        bars = self.db.get_daily_bars(
            start_date=start_date,
            end_date=end_ts.strftime("%Y-%m-%d"),
        )
        if bars is None or bars.empty:
            return pd.DataFrame()

        bars = bars[["date", "symbol", "close"]].copy()
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.sort_values(["symbol", "date"])
        bars["fwd_close"] = bars.groupby("symbol")["close"].shift(-forward_days)
        bars["fwd_ret"] = (
            bars["fwd_close"].astype("float64") / bars["close"].astype("float64") - 1.0
        )

        merged = wide.merge(
            bars[["date", "symbol", "fwd_ret"]],
            on=["date", "symbol"],
            how="left",
        )
        merged = merged.dropna(subset=["fwd_ret"])
        # 把因子列里的 inf 替换为 NaN(后续 IC 计算自动忽略)
        f_cols = [c for c in merged.columns if c not in ("date", "symbol", "fwd_ret")]
        for c in f_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged[f_cols] = merged[f_cols].replace([np.inf, -np.inf], np.nan)
        return merged

    # ---- 核心指标(全部向量化)------------------------------------------

    @staticmethod
    def _ic_panel(panel: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """向量化计算每日 IC,返回 DataFrame(index=date, columns=factor)。

        Spearman = Pearson(rank(x), rank(y))
        - 对每个 (date, factor) 计算因子排名与 fwd_ret 排名的 Pearson
        - 用 groupby(date) + transform("rank") 一次完成所有因子的截面排名
        """
        if panel.empty:
            return pd.DataFrame()
        df = panel[["date", "fwd_ret"] + factor_names].copy()
        # 对每日做截面排名:fwd_ret 一列、各因子一列
        ranks = df.groupby("date").transform(lambda s: s.rank())
        ranks["date"] = df["date"].values

        out: Dict[str, pd.Series] = {}
        for fn in factor_names:
            sub = ranks[["date", fn, "fwd_ret"]].dropna()
            # 截面少于 5 只样本的天直接舍弃
            counts = sub.groupby("date").size()
            valid_dates = counts[counts >= 5].index
            sub = sub[sub["date"].isin(valid_dates)]
            if sub.empty:
                out[fn] = pd.Series(dtype="float64")
                continue
            ic_by_date = sub.groupby("date").apply(
                lambda g: g[fn].corr(g["fwd_ret"])
            )
            out[fn] = ic_by_date
        return pd.DataFrame(out)

    @staticmethod
    def _group_metrics(
        panel: pd.DataFrame, factor: str, n_groups: int = 5
    ) -> tuple:
        """返回 (top_ret, bot_ret, monotonic) —— 分组收益与单调性。"""
        sub = panel[["date", factor, "fwd_ret"]].dropna()
        if sub.empty:
            return np.nan, np.nan, False

        def _qcut(s: pd.Series) -> pd.Series:
            try:
                return pd.qcut(s, n_groups, labels=False, duplicates="drop")
            except ValueError:
                return pd.Series(np.nan, index=s.index)

        sub = sub.copy()
        sub["group"] = sub.groupby("date")[factor].transform(_qcut)
        sub = sub.dropna(subset=["group"])
        if sub.empty:
            return np.nan, np.nan, False
        mean_by_group = sub.groupby("group")["fwd_ret"].mean().sort_index()
        if len(mean_by_group) < 2:
            return np.nan, np.nan, False
        bot_ret = float(mean_by_group.iloc[0])
        top_ret = float(mean_by_group.iloc[-1])
        diffs = mean_by_group.diff().dropna()
        monotonic = bool((diffs > 0).all() or (diffs < 0).all())
        return top_ret, bot_ret, monotonic

    @staticmethod
    def _turnover(
        panel: pd.DataFrame, factor: str, top_pct: float = 0.2
    ) -> float:
        """top 分位组日度换手率均值。"""
        sub = panel[["date", "symbol", factor]].dropna()
        if sub.empty:
            return np.nan

        prev_set: Optional[set] = None
        turnovers = []
        for dt, g in sub.groupby("date", sort=True):
            if len(g) < 5:
                continue
            cut = g[factor].quantile(1 - top_pct)
            cur_set = set(g.loc[g[factor] >= cut, "symbol"])
            if prev_set is not None and prev_set:
                overlap = len(prev_set & cur_set)
                turnovers.append(1 - overlap / len(prev_set))
            prev_set = cur_set
        return float(np.mean(turnovers)) if turnovers else np.nan

    # ---- 对外 API --------------------------------------------------------

    def evaluate_all(
        self,
        factor_names: Optional[List[str]] = None,
        start_date: str = None,
        end_date: str = None,
        forward_days: int = 5,
        n_groups: int = 5,
    ) -> pd.DataFrame:
        """批量评估,返回 summary DataFrame(按 |IR| 降序)。"""
        if factor_names is None:
            factor_names = self.db.list_factor_columns()
        if not factor_names:
            return pd.DataFrame(columns=EVAL_FIELDS)

        logger.info(
            "评估 %d 个因子,期间=%s~%s,forward_days=%d",
            len(factor_names), start_date, end_date, forward_days,
        )
        panel = self._load_panel(factor_names, start_date, end_date, forward_days)
        if panel.empty:
            return pd.DataFrame(columns=EVAL_FIELDS)

        # 一次性向量化算 IC 矩阵
        ic_df = self._ic_panel(panel, factor_names)

        rows: List[Dict] = []
        for fn in factor_names:
            ic_series = ic_df.get(fn, pd.Series(dtype="float64")).dropna()
            n_days = int(len(ic_series))
            if n_days < 1:
                rows.append({
                    "factor_name": fn,
                    "ic_mean": np.nan, "ic_std": np.nan, "ir": np.nan,
                    "ic_t_stat": np.nan, "turnover": np.nan,
                    "top_group_ret": np.nan, "bot_group_ret": np.nan,
                    "monotonic": False, "n_days": 0,
                })
                continue
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
            t_stat = (
                ic_mean / (ic_std / np.sqrt(n_days))
                if ic_std and ic_std > 0 and n_days > 1
                else np.nan
            )
            top_ret, bot_ret, monotonic = self._group_metrics(panel, fn, n_groups)
            turnover = self._turnover(panel, fn)
            rows.append({
                "factor_name": fn,
                "ic_mean": ic_mean, "ic_std": ic_std,
                "ir": float(ir) if pd.notna(ir) else np.nan,
                "ic_t_stat": float(t_stat) if pd.notna(t_stat) else np.nan,
                "turnover": turnover,
                "top_group_ret": top_ret, "bot_group_ret": bot_ret,
                "monotonic": monotonic, "n_days": n_days,
            })
            logger.debug(
                "  %-20s IR=%6.3f IC=%7.4f n=%4d",
                fn, rows[-1]["ir"] if pd.notna(rows[-1]["ir"]) else float("nan"),
                ic_mean, n_days,
            )

        out = pd.DataFrame(rows, columns=EVAL_FIELDS)
        out["abs_ir"] = out["ir"].abs()
        out = out.sort_values("abs_ir", ascending=False, na_position="last")
        return out.drop(columns=["abs_ir"])

    # ---- 写入 registry --------------------------------------------------

    def to_registry_records(
        self,
        summary: pd.DataFrame,
        category_map: Optional[Dict[str, str]] = None,
        description_map: Optional[Dict[str, str]] = None,
        ir_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """summary → factor_registry 落库格式,|IR| ≥ 阈值则 enabled=True。"""
        if summary is None or summary.empty:
            return pd.DataFrame()
        df = summary.copy()
        df["category"] = df["factor_name"].map(category_map or {}).fillna("技术")
        df["description"] = df["factor_name"].map(description_map or {}).fillna("")
        df["enabled"] = df["ir"].abs().fillna(0) >= ir_threshold
        return df[[
            "factor_name", "category", "description",
            "ic_mean", "ic_std", "ir", "ic_t_stat",
            "turnover", "top_group_ret", "bot_group_ret",
            "monotonic", "enabled",
        ]]