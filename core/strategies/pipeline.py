"""
统一策略回测管道 — 一行代码跑任意策略配置

提供 StrategyPipeline 类，封装数据加载/信号构建/回测/组合分析/窗口验证全流程。

用法:
    from core.strategies.pipeline import StrategyPipeline

    # 基于3D数组的手写信号管道
    p = StrategyPipeline(
        signal_builder=my_signal_fn,  # fn(z3, fi, nd, ns) -> np.ndarray
        name="my_strategy",
        rebal_freq=10, top_n=50, min_hold_days=5,
        positioner_type='rp',  # 'rp' 或 'covrp'
        tx_cost=0.0012,
    )
    result = p.run(start='2019-01-01', end='2026-04-30')
    windows = p.window_analysis()
    combo = p.combine(other_p, weight=0.4)
    p.export('my_strategy')

    # 从配置字典创建
    p = StrategyPipeline.from_config(config_dict)
"""
from __future__ import annotations
import os, json, logging
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from core.database import Database
from core.positioners import RPPortfolioWeights

logger = logging.getLogger(__name__)

WINDOWS = [
    ("2019修复牛", "2019-01-02", "2019-12-31"),
    ("2020疫情",   "2020-01-02", "2020-12-31"),
    ("2021结构牛", "2021-01-04", "2021-12-31"),
    ("2022熊市",   "2022-01-04", "2022-12-30"),
    ("2023震荡",   "2023-01-03", "2023-12-29"),
    ("2024反弹",   "2024-01-02", "2024-12-31"),
    ("2025至今",   "2025-01-02", "2026-04-30"),
]

DEFAULT_FACTORS = list(set([
    'a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position',
]))


@dataclass
class BacktestMetrics:
    """回测指标。"""
    name: str = ""
    window: str = "全区间"
    annual_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    n_days: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "window": self.window,
            "annual_return": round(self.annual_return, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "calmar": round(self.calmar, 4),
            "win_rate": round(self.win_rate, 4),
            "n_trades": self.n_trades,
            "n_days": self.n_days,
        }


class StrategyPipeline:
    """统一策略回测管道。

    封装完整回测流程: 数据加载 → 信号构建 → 回测 → 指标计算 → 窗口分析 → 导出。

    Parameters
    ----------
    signal_builder : callable, optional
        信号构建函数，签名为 fn(z3, fi, nd, ns) -> np.ndarray (nd×ns)
        若不提供，需在子类中覆盖 _build_signal()
    name : str
        策略名称
    rebal_freq : int
        调仓频率（天数）
    top_n : int
        每期选股数量上限
    min_hold_days : int
        最小持有天数
    positioner_type : str
        仓位分配器类型: 'rp' (风险平价) 或 'covrp' (协方差风险平价)
    tx_cost : float
        交易成本费率
    factor_names : list, optional
        因子列表，默认使用 DEFAULT_FACTORS
    """

    def __init__(
        self,
        signal_builder: Optional[Callable] = None,
        name: str = "strategy",
        rebal_freq: int = 3,
        top_n: int = 40,
        min_hold_days: int = 5,
        positioner_type: str = "rp",
        tx_cost: float = 0.0012,
        factor_names: Optional[List[str]] = None,
    ):
        self.signal_builder = signal_builder
        self.name = name
        self.rebal_freq = rebal_freq
        self.top_n = top_n
        self.min_hold_days = min_hold_days
        self.positioner_type = positioner_type
        self.tx_cost = tx_cost
        self.factor_names = factor_names or list(DEFAULT_FACTORS)

        self._data_loaded = False
        self._signal_built = False

        self.z3: Optional[np.ndarray] = None
        self.fwd: Optional[np.ndarray] = None
        self.dm: Optional[np.ndarray] = None
        self.tks: List[str] = []
        self.nd: int = 0
        self.ns: int = 0
        self.ds: List[pd.Timestamp] = []
        self.fi: Dict[str, int] = {}

        self.sig: Optional[np.ndarray] = None
        self._dr: Optional[np.ndarray] = None
        self._last_result: Optional[BacktestMetrics] = None
        self._window_results: List[BacktestMetrics] = []

    # ──────────────────────────────────────────
    # 数据加载
    # ──────────────────────────────────────────
    def load(self, start_date: str = "2018-01-01", end_date: str = "2026-04-30"):
        """加载数据并构建3D因子矩阵。

        Parameters
        ----------
        start_date, end_date : str
            数据加载区间（需覆盖回测区间）
        """
        db = Database()
        df = db.get_factors(
            start_date=start_date, end_date=end_date,
            factor_names=self.factor_names, with_close=True
        )
        df['date'] = pd.to_datetime(df['date'])
        ds = sorted(df['date'].unique())
        tks = db.get_symbols()['symbol'].tolist()
        nd, ns, nf = len(ds), len(tks), len(self.factor_names)
        t2i = {t: i for i, t in enumerate(tks)}
        d2i = {d: i for i, d in enumerate(ds)}

        v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32)
        dm = np.zeros((nd, ns), dtype=bool)
        cl = np.zeros((nd, ns), dtype=np.float32)

        di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
        si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
        v = si >= 0
        di, si = di[v], si[v]

        for fi, fc in enumerate(self.factor_names):
            if fc in df.columns:
                v3[di, si, fi] = df[fc].values[v].astype(np.float32)
        cl[di, si] = df['close'].values[v].astype(np.float32)
        dm[di, si] = True
        np.nan_to_num(v3, nan=0.0, copy=False)
        np.nan_to_num(cl, nan=0.0, copy=False)

        fwd = np.zeros((nd, ns), dtype=np.float32)
        for d in range(nd - 1):
            b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
            fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

        z3 = np.zeros_like(v3)
        for fi in range(nf):
            a = v3[:, :, fi]
            for d in range(nd):
                r = a[d, :]
                nz = r[r != 0]
                if len(nz) > 1:
                    lo, hi = np.quantile(nz, [0.01, 0.99])
                    c = np.clip(r, lo, hi)
                    mu, sd = np.mean(c), np.std(c)
                    z3[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0

        self.z3 = z3
        self.fwd = fwd
        self.dm = dm
        self.tks = tks
        self.nd = nd
        self.ns = ns
        self.ds = ds
        self.fi = {fn: i for i, fn in enumerate(self.factor_names)}
        self._data_loaded = True

        logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
        return self

    # ──────────────────────────────────────────
    # 信号构建
    # ──────────────────────────────────────────
    def build_signal(self) -> np.ndarray:
        """构建信号矩阵。"""
        if self.signal_builder is not None:
            self.sig = self.signal_builder(self.z3, self.fi, self.nd, self.ns)
        else:
            self.sig = self._build_signal()
        self._signal_built = True
        return self.sig

    def _build_signal(self) -> np.ndarray:
        """默认信号构建（等权因子合成），子类可覆盖。"""
        w = np.ones(len(self.factor_names), dtype=np.float32)
        w /= len(w)
        sig = np.nan_to_num(
            np.tensordot(self.z3, w, axes=(2, 0)),
            nan=-1e10, neginf=-1e10
        )
        return sig

    # ──────────────────────────────────────────
    # 回测
    # ──────────────────────────────────────────
    def run(self, start: Optional[str] = None, end: Optional[str] = None) -> BacktestMetrics:
        """执行全区间回测。

        Parameters
        ----------
        start, end : str, optional
            回测区间，默认使用全部数据
        """
        if not self._data_loaded:
            self.load()
        if not self._signal_built:
            self.build_signal()

        sig = self.sig
        fwd = self.fwd
        dm = self.dm
        nd = self.nd

        if start or end:
            sidx, eidx = 0, nd
            if start:
                sd = pd.Timestamp(start)
                sidx = next((i for i, d in enumerate(self.ds) if d >= sd), 0)
            if end:
                ed = pd.Timestamp(end)
                eidx = next((i for i, d in enumerate(self.ds) if d > ed), nd)
            sig = self.sig[sidx:eidx]
            fwd = self.fwd[sidx:eidx]
            dm = self.dm[sidx:eidx]
            nd = eidx - sidx

        dr, nt = self._backtest(sig, fwd, dm, nd)
        self._dr = dr

        metrics = self._calc_metrics(dr, nd, self.name, "全区间", nt)
        self._last_result = metrics
        return metrics

    def _backtest(
        self, sig: np.ndarray, fwd: np.ndarray, dm: np.ndarray, nd: int
    ) -> Tuple[np.ndarray, int]:
        """执行回测核心逻辑。"""
        ns = sig.shape[1]

        if self.positioner_type == 'covrp':
            return self._backtest_covrp(sig, fwd, dm, nd, ns)
        return self._backtest_rp(sig, fwd, dm, nd, ns)

    def _backtest_rp(
        self, sig: np.ndarray, fwd: np.ndarray, dm: np.ndarray, nd: int, ns: int
    ) -> Tuple[np.ndarray, int]:
        """RP风险平价回测。"""
        alloc = RPPortfolioWeights(top_n=self.top_n, min_hold_days=self.min_hold_days)
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0
        dr = np.zeros(nd, dtype=np.float64)
        nt = 0

        for i in range(1, nd):
            rebal = (i % self.rebal_freq == 0)
            txc = 0.0
            if rebal:
                nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost
                if to > 0.01:
                    nt += 1
                pw = nw
                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0:
                        hs[j] = rh + 1
            else:
                mk = dm[i] & (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(ns, dtype=np.float32)
                    pw[mk] = p2

            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1

        return dr, nt

    def _backtest_covrp(
        self, sig: np.ndarray, fwd: np.ndarray, dm: np.ndarray, nd: int, ns: int
    ) -> Tuple[np.ndarray, int]:
        """协方差风险平价回测。"""
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0
        dr = np.zeros(nd, dtype=np.float64)
        nt = 0

        for i in range(1, nd):
            rebal = (i % self.rebal_freq == 0)
            txc = 0.0
            if rebal:
                si = np.argsort(-sig[i])[:self.top_n]
                if i >= 20:
                    seg = fwd[max(0, i - 20):i, :]
                    sub = seg[:, si]
                    sub = sub[:, ~np.any(np.isnan(sub) | np.isinf(sub), axis=0)]
                    if sub.shape[1] >= 2:
                        try:
                            cov = np.cov(sub.T)
                            iv = 1.0 / np.sqrt(np.diag(cov) + 1e-10)
                        except Exception:
                            iv = np.ones(sub.shape[1])
                    else:
                        iv = np.ones(sub.shape[1])
                else:
                    iv = np.ones(min(self.top_n, ns))

                nw = np.zeros(ns)
                sidx = si[:len(iv)]
                if len(sidx) > 0:
                    nw[sidx] = iv / np.sum(iv)

                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost
                if to > 0.01:
                    nt += 1
                pw = nw
                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0:
                        hs[j] = rh + 1
            else:
                mk = dm[i] & (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(ns, dtype=np.float32)
                    pw[mk] = p2

            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1

        return dr, nt

    # ──────────────────────────────────────────
    # 指标计算
    # ──────────────────────────────────────────
    def _calc_metrics(
        self, dr: np.ndarray, nd: int, name: str, wname: str, nt: int
    ) -> BacktestMetrics:
        eq = np.ones(nd)
        for i in range(1, nd):
            eq[i] = eq[i - 1] * (1.0 + dr[i])
        tr = float(eq[-1] / eq[0] - 1.0)
        ny = nd / 252.0
        ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
        lr = np.log(eq[1:] / eq[:-1])
        sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
        cm = np.maximum.accumulate(eq)
        dd = (eq - cm) / cm
        mdd = float(np.min(dd))
        cal = ar / abs(mdd) if abs(mdd) > 0 else 0
        wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)

        logger.info(
            f"  [{name}][{wname}] 年化={ar*100:.2f}% "
            f"Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}"
        )
        return BacktestMetrics(
            name=name, window=wname,
            annual_return=ar, sharpe=sp, max_drawdown=mdd,
            calmar=cal, win_rate=wr, n_trades=nt, n_days=nd,
        )

    # ──────────────────────────────────────────
    # 窗口分析
    # ──────────────────────────────────────────
    def window_analysis(
        self, windows: Optional[List[Tuple[str, str, str]]] = None
    ) -> List[BacktestMetrics]:
        """对全区间回测做分窗口分析。

        Parameters
        ----------
        windows : list of (name, start, end), optional
            默认使用预定义7窗口

        Returns
        -------
        list[BacktestMetrics]
            每个窗口的指标（第一个为全区间）
        """
        if self._dr is None:
            self.run()

        dr = self._dr
        nd = len(dr)
        total = self._calc_metrics(dr, nd, self.name, "全区间", 0)
        results = [total]

        windows = windows or WINDOWS
        for wname, ws, we in windows:
            ws_d = pd.Timestamp(ws)
            we_d = pd.Timestamp(we)
            in_window = [j for j, d in enumerate(self.ds) if ws_d <= d <= we_d]
            if len(in_window) < 5:
                results.append(BacktestMetrics(
                    name=self.name, window=wname,
                    n_days=len(in_window),
                ))
                continue

            w_start = in_window[0]
            w_end = in_window[-1] + 1
            w_dr = dr[w_start:w_end]
            w_dr[0] = 0.0
            wr = self._calc_metrics(w_dr, len(w_dr), self.name, wname, 0)
            wr.n_days = len(in_window)
            results.append(wr)

        self._window_results = results
        return results

    # ──────────────────────────────────────────
    # 组合分析
    # ──────────────────────────────────────────
    def get_return_series(self) -> np.ndarray:
        """获取日收益率序列。"""
        if self._dr is None:
            self.run()
        return self._dr.copy()

    def combine(
        self, other: StrategyPipeline, weight: float = 0.5
    ) -> BacktestMetrics:
        """与另一个策略组合。

        Parameters
        ----------
        other : StrategyPipeline
            另一个策略实例
        weight : float
            本策略的权重（对方权重为 1-weight）

        Returns
        -------
        BacktestMetrics
            组合后的指标
        """
        dr1 = self.get_return_series()
        dr2 = other.get_return_series()
        nd = min(len(dr1), len(dr2))
        dr = dr1[:nd] * weight + dr2[:nd] * (1.0 - weight)
        name = f"{self.name}({weight:.1f})×{other.name}({1-weight:.1f})"
        metrics = self._calc_metrics(dr, nd, name, "组合", 0)
        return metrics

    @staticmethod
    def combo_from_series(
        dr1: np.ndarray, dr2: np.ndarray,
        dr3: Optional[np.ndarray] = None,
        w1: float = 0.5, w2: float = 0.5, w3: float = 0.0,
    ) -> dict:
        """直接对收益率序列做组合分析（无需Pipeline实例）。"""
        nd = len(dr1)
        dr = dr1 * w1 + dr2 * w2 + (dr3 * w3 if dr3 is not None else 0)
        eq = np.ones(nd)
        for i in range(1, nd):
            eq[i] = eq[i - 1] * (1.0 + dr[i])

        tr = float(eq[-1] / eq[0] - 1.0)
        ny = nd / 252.0
        ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
        lr = np.log(eq[1:] / eq[:-1])
        sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
        cm = np.maximum.accumulate(eq)
        dd = (eq - cm) / cm
        mdd = float(np.min(dd))
        cal = ar / abs(mdd) if abs(mdd) > 0 else 0
        wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)

        return {
            "annual_return": round(ar, 4),
            "sharpe": round(sp, 4),
            "max_drawdown": round(mdd, 4),
            "calmar": round(cal, 4),
            "win_rate": round(wr, 4),
            "n_trades": 0,
        }

    # ──────────────────────────────────────────
    # 导出
    # ──────────────────────────────────────────
    def export(self, folder_name: str, root_dir: Optional[str] = None):
        """导出策略配置和回测结果到策略文件夹。

        Parameters
        ----------
        folder_name : str
            文件夹名称
        root_dir : str, optional
            根目录，默认 daily/{today}/results
        """
        if root_dir is None:
            today = datetime.now().strftime("%Y-%m-%d")
            root_dir = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "daily", today, "results"
            )

        out_dir = os.path.join(root_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        config = {
            "name": self.name,
            "rebal_freq": self.rebal_freq,
            "top_n": self.top_n,
            "min_hold_days": self.min_hold_days,
            "positioner_type": self.positioner_type,
            "tx_cost": self.tx_cost,
            "n_factors": len(self.factor_names),
        }

        if self._last_result is not None:
            config["results"] = self._last_result.to_dict()

        if self._window_results:
            config["window_analysis"] = [w.to_dict() for w in self._window_results]

        out_path = os.path.join(out_dir, "config.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"已导出至: {out_path}")
        return out_path

    # ──────────────────────────────────────────
    # 类方法：从配置创建
    # ──────────────────────────────────────────
    @classmethod
    def from_config(cls, config: dict) -> StrategyPipeline:
        """从配置字典创建管道实例。

        Parameters
        ----------
        config : dict
            配置字典，支持字段:
            - signal_builder: callable
            - name: str
            - rebal_freq: int
            - top_n: int
            - min_hold_days: int
            - positioner_type: str ('rp'|'covrp')
            - tx_cost: float
            - factor_names: list[str]
            - selector_type: str (仅用于文档参考)

        Returns
        -------
        StrategyPipeline
        """
        known_keys = {
            'signal_builder', 'name', 'rebal_freq', 'top_n',
            'min_hold_days', 'positioner_type', 'tx_cost', 'factor_names',
        }
        pipe_kwargs = {k: v for k, v in config.items() if k in known_keys}
        return cls(**pipe_kwargs)

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────
    def summary(self) -> str:
        """返回可打印的结果摘要。"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"策略: {self.name}")
        lines.append(f"配置: rf={self.rebal_freq} tn={self.top_n} "
                      f"mhd={self.min_hold_days} tx={self.tx_cost}")
        lines.append("-" * 60)

        if self._last_result:
            r = self._last_result
            lines.append(f"全区间: 年化={r.annual_return*100:.2f}% "
                          f"Sharpe={r.sharpe:.3f} 回撤={r.max_drawdown*100:.2f}% "
                          f"Calmar={r.calmar:.3f}")

        if self._window_results:
            lines.append("-" * 60)
            lines.append(f"{'窗口':<12} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}")
            for w in self._window_results:
                if w.n_days == 0:
                    continue
                lines.append(
                    f"{w.window:<12} {w.annual_return*100:>+6.2f}% "
                    f"{w.sharpe:>7.3f} {w.max_drawdown*100:>6.1f}% "
                    f"{w.calmar:>7.3f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"StrategyPipeline(name='{self.name}', rf={self.rebal_freq}, "
            f"tn={self.top_n}, positioner='{self.positioner_type}')"
        )
