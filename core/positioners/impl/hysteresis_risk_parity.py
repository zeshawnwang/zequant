"""
带迟滞效应的仓位分配器

提供两个级别的分配器：
- HysteresisAllocator: 底层数组级分配器，直接操作 numpy 权重向量，
  适用于 GA 快评 / 回测引擎内部调用
- HysteresisRiskParityBuilder: 高层信号级分配器，实现 IPortfolioBuilder
  接口，适用于策略引擎 generate_orders() 调用

迟滞机制：
1. 大仓位惯性：持仓超过阈值且调整幅度过小时跳过
2. 调整排序截断：只执行调整幅度前若干比例的调仓明细
"""
from typing import Dict, List, Optional
import numpy as np
from core.positioners.impl.risk_parity import RiskParityBuilder, SignalType


class HysteresisAllocator:
    """底层数组级分配器，可选迟滞过滤。

    当 enable_hysteresis=True 时，在风险平价权重上增加三道过滤：
    1. **大仓位惯性**: 持仓 > large_pos_threshold 的股票，调整幅度
       必须 > min_adjust_delta 才执行
    2. **调整排序截断**: 按 |调整差| 从大到小排序，只执行前 keep_ratio 比例
    3. **重归一化**: 截断后权重之和 ≠ 1.0，重新缩放到满仓

    当 enable_hysteresis=False 时，行为等价于原始风险平价分配器。

    参数 (可根据实盘效果微调):
        enable_hysteresis: 是否启用迟滞过滤。默认 True 保持向后兼容。
            设为 False 时退化为基础风险平价分配器
        top_n: 最多选股数
        min_hold_days: 最低持仓天数 (锁仓)
        large_pos_threshold: 大仓位认定阈值, >此值受惯性约束
        min_adjust_delta: 大仓位最小调整幅度, |delta|<此值则不调
        keep_ratio: 保留前百分之多少的大调整 (0~1)
        vol_lookback: 波动率回溯天数 (风险平价用)
    """

    def __init__(
        self,
        enable_hysteresis: bool = True,
        top_n: int = 30,
        min_hold_days: int = 5,
        large_pos_threshold: float = 0.10,
        min_adjust_delta: float = 0.02,
        keep_ratio: float = 0.70,
        vol_lookback: int = 20,
    ):
        self.enable_hysteresis = enable_hysteresis
        self.top_n = top_n
        self.min_hold_days = min_hold_days
        self.large_pos_threshold = large_pos_threshold
        self.min_adjust_delta = min_adjust_delta
        self.keep_ratio = keep_ratio
        self.vol_lookback = vol_lookback

    def allocate(
        self,
        scores: np.ndarray,
        fwd_ret: np.ndarray,
        t: int,
        prev_weights: np.ndarray,
        hold_since: np.ndarray,
    ) -> np.ndarray:
        """计算调仓后权重。

        Args:
            scores: 因子综合得分 (n_symbols,)
            fwd_ret: 前向收益 (n_dates, n_symbols)
            t: 当前时间索引
            prev_weights: 上一期权重 (n_symbols,)
            hold_since: 每只股票上次买入时间, -1=未持有

        Returns:
            new_weights: 调仓后权重 (n_symbols,)
        """
        n_symbols = len(scores)
        valid = ~np.isnan(scores)

        # ---- 1. 锁仓(最低持有) ----
        locked = np.zeros(n_symbols, dtype=bool)
        for i in range(n_symbols):
            if hold_since[i] > 0 and (t - hold_since[i]) < self.min_hold_days and prev_weights[i] > 0:
                locked[i] = True
        locked_weight = float(np.sum(prev_weights[locked]))

        # ---- 2. 风险平价目标权重 ----
        available = valid & ~locked
        n_avail = int(np.sum(available))
        if n_avail < 1:
            return prev_weights.copy()

        avail_scores = scores.copy()
        avail_scores[~available] = -np.inf
        n_pick = min(self.top_n, n_avail)
        best = np.argpartition(-avail_scores, n_pick)[:n_pick]

        if t >= self.vol_lookback:
            hist_ret = fwd_ret[t - self.vol_lookback : t, :]
            vol = np.nanstd(hist_ret, axis=0) + 1e-10
            vol = np.nan_to_num(vol, nan=1e10, posinf=1e10, neginf=1e10)
            inv_vol = 1.0 / vol
        else:
            inv_vol = np.ones(n_symbols, dtype=np.float32)

        rp_w = inv_vol[best] / max(np.sum(inv_vol[best]), 1e-10) * (1.0 - locked_weight)

        target_weights = np.zeros(n_symbols, dtype=np.float32)
        target_weights[best] = rp_w
        target_weights[locked] = prev_weights[locked]

        # ---- 3. 计算调整差 delta ----
        delta = target_weights - prev_weights

        if self.enable_hysteresis:
            # ---- 4. 大仓位惯性过滤 ----
            large_mask = prev_weights >= self.large_pos_threshold
            small_delta = np.abs(delta) < self.min_adjust_delta
            delta[large_mask & small_delta] = 0.0

            # ---- 5. 调整排序截断 ----
            nonzero_idx = np.where(np.abs(delta) > 1e-10)[0]
            if len(nonzero_idx) > 0:
                abs_deltas = np.abs(delta[nonzero_idx])
                n_keep = max(1, int(len(nonzero_idx) * self.keep_ratio))
                top_k_indices = np.argpartition(-abs_deltas, n_keep)[:n_keep]
                skip_idx = np.setdiff1d(nonzero_idx, nonzero_idx[top_k_indices])
                delta[skip_idx] = 0.0

        # ---- 6. 计算新权重并归一化 ----
        new_weights = prev_weights + delta
        w_sum = np.sum(new_weights)
        if w_sum > 1e-10:
            new_weights = new_weights / w_sum
        else:
            return prev_weights.copy()

        return new_weights

    def __repr__(self) -> str:
        return (
            f"HysteresisAllocator(top_n={self.top_n}, "
            f"large_pos_threshold={self.large_pos_threshold}, "
            f"min_adjust_delta={self.min_adjust_delta}, "
            f"keep_ratio={self.keep_ratio})"
        )


class HysteresisRiskParityBuilder(RiskParityBuilder):
    """带迟滞效应的风险平价仓位分配器。

    在 RiskParityBuilder 风险平价权重基础上，增加调仓过滤：
    - 大仓位惯性：对持仓 > large_pos_threshold 的股票，调整幅度小于
      min_adjust_delta 时不做调整
    - 排序截断：仅保留调整差绝对值 top keep_ratio 比例的调仓明细
    """

    def __init__(self,
                 large_pos_threshold: float = 0.10,
                 min_adjust_delta: float = 0.02,
                 keep_ratio: float = 0.70,
                 volatility_factor: str = 'volatility_20',
                 max_weight: float = 0.15,
                 reserve_cash_ratio: float = 0.1,
                 lookback_window: int = 60,
                 min_history: int = 20):
        """
        Args:
            large_pos_threshold: 大仓位阈值，持仓超过该比例的股票受惯性保护
            min_adjust_delta: 最小调整阈值，大仓位的调整幅度小于此值时跳过
            keep_ratio: 保留比例，按 |调整差| 降序保留前 keep_ratio 的调仓
            volatility_factor: 波动率因子名（回退方案使用）
            max_weight: 单股最大权重
            reserve_cash_ratio: 保留现金比例
            lookback_window: 计算协方差的历史窗口天数
            min_history: 最小历史数据要求
        """
        super().__init__(
            volatility_factor=volatility_factor,
            max_weight=max_weight,
            reserve_cash_ratio=reserve_cash_ratio,
            lookback_window=lookback_window,
            min_history=min_history,
        )
        self.large_pos_threshold = large_pos_threshold
        self.min_adjust_delta = min_adjust_delta
        self.keep_ratio = keep_ratio

    def allocate(self, signals, total_cash: float,
                 current_positions: Dict) -> Dict[str, int]:
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        if not buy_signals:
            return {}

        n = len(buy_signals)
        symbols = [s.symbol for s in buy_signals]

        # 1. 计算目标权重（同父类）
        target_w = self._compute_risk_parity_weights(symbols)
        if target_w is None:
            target_w = self._inverse_vol_weights(buy_signals)

        target_w = np.minimum(target_w, self.max_weight)
        target_w = target_w / target_w.sum()

        # 2. 计算当前组合总资产
        #    总权益 = 可用现金 + 所有持仓市值
        total_equity = total_cash
        for sig in buy_signals:
            pos = current_positions.get(sig.symbol)
            if pos:
                total_equity += pos.quantity * sig.price

        # 3. 计算各标的当前权重
        current_w = np.zeros(n, dtype=np.float64)
        for i, sig in enumerate(buy_signals):
            pos = current_positions.get(sig.symbol)
            if pos and total_equity > 0:
                current_w[i] = (pos.quantity * sig.price) / total_equity

        # 4. 计算调整差 = 目标权重 - 当前权重
        delta = target_w - current_w

        # 5. 迟滞过滤
        # 5a. 大仓位惯性：持仓 > large_pos_threshold 且 |调整| < min_adjust_delta 则跳过
        for i in range(n):
            if current_w[i] >= self.large_pos_threshold \
                    and abs(delta[i]) < self.min_adjust_delta:
                delta[i] = 0.0

        # 5b. 排序截断：按 |调整差| 降序，保留前 keep_ratio
        abs_delta = np.abs(delta)
        n_keep = max(1, int(n * self.keep_ratio))
        if n_keep < n:
            cutoff = np.sort(abs_delta)[-n_keep]
            delta[abs_delta < cutoff] = 0.0

        # 6. 新权重 = 当前权重 + 过滤后的调整差
        new_w = current_w + delta
        new_w = np.maximum(new_w, 0.0)
        if new_w.sum() > 1e-10:
            new_w = new_w / new_w.sum()
        else:
            new_w = target_w

        # 7. 转为买入股数
        usable_cash = total_cash * (1 - self.reserve_cash_ratio)
        allocation = {}

        for i, sig in enumerate(buy_signals):
            target_value = new_w[i] * total_equity
            current_value = current_w[i] * total_equity
            additional = target_value - current_value

            if additional > 0:
                shares = int(additional / sig.price / 100) * 100
                shares = min(shares, int(usable_cash / sig.price / 100) * 100)
                if shares >= 100:
                    allocation[sig.symbol] = shares
                    usable_cash -= shares * sig.price

        return allocation
