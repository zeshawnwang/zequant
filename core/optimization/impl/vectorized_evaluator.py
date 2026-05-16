"""
向量化回测评估器 — V1核心的组件化封装。

快速评估一组因子权重在历史数据上的表现。
使用纯 numpy 向量化运算，避免 BacktestEngine 的事件驱动开销。

典型用法:
    evaluator = VectorizedEvaluator(tx_cost_rate=0.0012)
    result = evaluator.evaluate(weights, factor_z, fwd_ret, rebal_freq=3)
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
import numpy as np

from core.positioners import RPPortfolioWeights


@dataclass
class EvalResult:
    """一次回测评估的结果。"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    turnover_count: int = 0
    avg_turnover: float = 0.0
    n_trades: int = 0
    total_tx_cost: float = 0.0
    equity_curve: list = field(default_factory=list)

    def composite_score(self) -> float:
        return (self.annual_return * 0.30 + self.sharpe * 0.25 +
                self.calmar * 0.30 + self.win_rate * 0.15)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, (list, np.ndarray)) or k == "equity_curve"}


class VectorizedEvaluator:
    """向量化回测评估器。

    给定因子权重，在全历史数据上做周频/日频调仓、风险平价分配，
    输出完整绩效指标。

    Args:
        tx_cost_rate: 单边交易成本率
        portfolio_builder: 仓位分配器，默认 RPPortfolioWeights(top_n=40, min_hold_days=5)
    """

    def __init__(self, tx_cost_rate: float = 0.0012,
                 portfolio_builder: Optional[RPPortfolioWeights] = None):
        self.tx_cost_rate = tx_cost_rate
        self.portfolio = portfolio_builder or RPPortfolioWeights(top_n=40, min_hold_days=5)

    def evaluate(self, weights: np.ndarray, factor_z: np.ndarray,
                 fwd_ret: np.ndarray, data_mask: np.ndarray = None,
                 rebal_freq: int = 3, top_n: int = 40,
                 return_equity: bool = False) -> EvalResult:
        n_dates, n_sym = factor_z.shape[:2]
        composite = np.dot(factor_z, weights.astype(np.float32))
        pw = np.zeros(n_sym, dtype=np.float32)
        hs = np.full(n_sym, -1, dtype=np.int32)
        rh = 0
        eq = np.ones(n_dates, dtype=np.float64)
        dr = np.zeros(n_dates, dtype=np.float64)
        ttx = 0.0
        nt = 0
        total_to = 0.0
        rebal_days = 0

        for i in range(1, n_dates):
            rebal = (i % rebal_freq == 0)
            if rebal:
                nw = self.portfolio.allocate(composite[i], fwd_ret, i, pw, hs, rh)
                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost_rate
                ttx += txc
                total_to += to
                rebal_days += 1
                if to > 0.01:
                    nt += 1
                pw = nw
                for j in range(n_sym):
                    if nw[j] > 0 and hs[j] < 0:
                        hs[j] = rh + 1
            else:
                mk = (data_mask[i] & (pw > 0)) if data_mask is not None else (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(n_sym, dtype=np.float32)
                    pw[mk] = p2

            rt = float(np.dot(pw, fwd_ret[i]))
            rt = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
            if rebal and to > 0.01:
                rt -= txc
            dr[i] = rt
            eq[i] = eq[i - 1] * (1.0 + rt)
            rh += 1

        tr = float(eq[-1] / eq[0] - 1.0)
        ny = n_dates / 252.0
        ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
        lr = np.log(eq[1:] / eq[:-1])
        sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
        cm = np.maximum.accumulate(eq)
        dd = (eq - cm) / cm
        mdd = float(np.min(dd))
        cal = ar / abs(mdd) if abs(mdd) > 0 else 0
        wd = int(np.sum(dr > 0))
        ld = int(np.sum(dr < 0))
        wr = wd / max(wd + ld, 1)
        avg_to = total_to / max(rebal_days, 1)

        result = EvalResult(
            total_return=tr, annual_return=ar, sharpe=sp,
            max_drawdown=mdd, calmar=cal, win_rate=wr,
            turnover_count=rebal_days, avg_turnover=avg_to,
            n_trades=nt, total_tx_cost=ttx,
        )
        if return_equity:
            result.equity_curve = eq.tolist()
        return result

    def evaluate_with_risk_check(self, weights: np.ndarray, factor_z: np.ndarray,
                                  fwd_ret: np.ndarray, data_mask: np.ndarray = None,
                                  rebal_freq: int = 3, top_n: int = 40,
                                  l1_lambda: float = 0.0, turnover_penalty: float = 0.0,
                                  risk_constraints=None) -> float:
        result = self.evaluate(weights, factor_z, fwd_ret, data_mask,
                               rebal_freq, top_n)

        if risk_constraints is not None:
            rc = risk_constraints.check_backtest_result(
                annual_return=result.annual_return,
                max_drawdown=result.max_drawdown,
                volatility=0.3,
                calmar_ratio=result.calmar,
                win_rate=result.win_rate,
            )
            if not rc.passed:
                return -1.0

        score = result.composite_score()
        l1_pen = l1_lambda * float(np.sum(np.abs(weights)))
        to_pen = turnover_penalty * max(0, result.avg_turnover - 0.5)
        return float(score - l1_pen - to_pen)
