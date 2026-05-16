"""
RPPortfolioWeights — 纯 numpy 版风险平价仓位分配器。

与 core/positioners/impl/risk_parity.py 的区别：
  - RiskParityBuilder: 实现 IPortfolioBuilder 接口，接收 Signal 对象用于完整策略引擎
  - RPPortfolioWeights: 底层数组级分配器，直接操作 numpy 权重向量，
    适用于 GA 评估/内部回测引擎等场景（无需 Signal 对象）

用法:
    allocator = RPPortfolioWeights(top_n=40, min_hold_days=5)
    new_weights = allocator.allocate(scores, fwd_ret, t, prev_w, hold_since, rh_len)
"""
from __future__ import annotations
import numpy as np


class RPPortfolioWeights:
    """风险平价仓位分配（纯 numpy 版）。

    按因子综合得分排名选股，然后以波动率倒数加权分配仓位，
    同时支持最低持仓天数锁定。

    Args:
        top_n: 最多选股数
        min_hold_days: 最低持仓天数（锁定已持有但时间不足的股票）
        vol_lookback: 波动率回溯天数
    """

    def __init__(self, top_n: int = 30, min_hold_days: int = 5,
                 vol_lookback: int = 20):
        self.top_n = top_n
        self.min_hold_days = min_hold_days
        self.vol_lookback = vol_lookback

    def allocate(self, scores: np.ndarray, fwd_ret: np.ndarray, t: int,
                 prev_weights: np.ndarray, hold_since: np.ndarray,
                 rh_len: int) -> np.ndarray:
        """计算调仓后权重向量。

        Args:
            scores: 因子综合得分 (n_symbols,)
            fwd_ret: 前向收益矩阵 (n_dates, n_symbols)
            t: 当前时间索引
            prev_weights: 上一期持仓权重 (n_symbols,)
            hold_since: 每只股票上次买入时间索引, -1=未持有
            rh_len: 当前回测天数计数

        Returns:
            new_weights: 调仓后权重 (n_symbols,)
        """
        n_sym = len(scores)
        locked = np.zeros(n_sym, dtype=bool)
        for j in range(n_sym):
            if (hold_since[j] > 0 and prev_weights[j] > 0
                    and (rh_len - hold_since[j]) < self.min_hold_days):
                locked[j] = True
        locked_w = float(np.sum(prev_weights[locked]))

        sidx = np.argsort(-scores)[:self.top_n]
        vm = np.zeros(n_sym, dtype=bool)
        vm[sidx] = True
        avail = vm & ~locked
        aidx = np.where(avail)[0]
        if len(aidx) == 0:
            aidx = np.where(vm)[0]
        remaining = 1.0 - locked_w
        if remaining <= 0.0:
            return prev_weights.copy()

        if t >= self.vol_lookback:
            vol = np.nanstd(fwd_ret[max(0, t - self.vol_lookback):t], axis=0) + 1e-10
        else:
            vol = np.ones(n_sym, dtype=np.float32)
        iv = 1.0 / vol[aidx]
        ivs = float(np.sum(iv))
        tgt = (iv / ivs) * remaining if ivs > 0 else np.ones(len(aidx)) * remaining / len(aidx)

        nw = np.zeros(n_sym, dtype=np.float32)
        nw[locked] = prev_weights[locked]
        nw[aidx] = tgt
        return nw

    def __repr__(self):
        return f"RPPortfolioWeights(top_n={self.top_n}, min_hold={self.min_hold_days}d)"
