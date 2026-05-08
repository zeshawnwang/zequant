"""
风险平价仓位分配器（True Risk Parity）

基于协方差矩阵，使每只股票对组合风险的贡献相等。

数学原理:
    - 组合风险: σ_p = sqrt(w^T Σ w)
    - 资产i的风险贡献: RC_i = w_i * (Σw)_i / σ_p
    - 风险平价条件: RC_i = RC_j 对所有 i,j
    - 即: w_i * (Σw)_i = 常数 (对所有i相同)

求解方法:
    使用 Cyclical Coordinate Descent 迭代算法:
    w_i^{new} = sqrt(w_i * portfolio_var / (n * (Σw)_i))
    然后归一化。重复直到收敛。

回退方案:
    当协方差矩阵奇异或历史数据不足时，回退到逆波动率权重。
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .base import IPortfolioBuilder


class RiskParityBuilder(IPortfolioBuilder):
    """
    真实风险平价仓位分配器。

    通过历史收益率计算协方差矩阵，求解使各资产风险贡献相等的权重。
    若数据不足或协方差矩阵奇异，自动回退到逆波动率权重。
    """

    def __init__(self,
                 volatility_factor: str = 'volatility_20',
                 max_weight: float = 0.15,
                 reserve_cash_ratio: float = 0.1,
                 lookback_window: int = 60,
                 min_history: int = 20):
        """
        Args:
            volatility_factor: 波动率因子名（回退方案使用）
            max_weight: 单股最大权重
            reserve_cash_ratio: 保留现金比例
            lookback_window: 计算协方差的历史窗口天数
            min_history: 最小历史数据要求
        """
        self.volatility_factor = volatility_factor
        self.max_weight = max_weight
        self.reserve_cash_ratio = reserve_cash_ratio
        self.lookback_window = lookback_window
        self.min_history = min_history
        self._factor_data: Optional[pd.DataFrame] = None

    def set_factor_data(self, factor_data: pd.DataFrame):
        """接收历史因子数据，用于计算协方差矩阵。

        由 QuantStrategy.generate_orders() 在调用 allocate() 之前自动传入。
        """
        self._factor_data = factor_data.copy() if factor_data is not None else None

    def allocate(self, signals, total_cash: float,
                 current_positions: Dict) -> Dict[str, int]:
        from core.strategy import SignalType
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        if not buy_signals:
            return {}

        symbols = [s.symbol for s in buy_signals]

        # 尝试使用协方差矩阵计算真实风险平价权重
        weights = self._compute_risk_parity_weights(symbols)

        # 如果计算失败，回退到逆波动率权重
        if weights is None:
            weights = self._inverse_vol_weights(buy_signals)

        # 限制最大权重
        weights = np.minimum(weights, self.max_weight)
        weights = weights / weights.sum()  # 重新归一化

        usable_cash = total_cash * (1 - self.reserve_cash_ratio)
        allocation = {}
        for i, sig in enumerate(buy_signals):
            stock_cash = usable_cash * weights[i]
            shares = int(stock_cash / sig.price / 100) * 100
            if shares >= 100:
                allocation[sig.symbol] = shares

        return allocation

    def _compute_risk_parity_weights(self, symbols: List[str]) -> Optional[np.ndarray]:
        """计算真实风险平价权重，失败返回 None。"""
        if self._factor_data is None or self._factor_data.empty:
            return None

        df = self._factor_data[self._factor_data['symbol'].isin(symbols)].copy()
        if df.empty or 'close' not in df.columns or 'date' not in df.columns:
            return None

        df['date'] = pd.to_datetime(df['date'])

        # Pivot 成宽表: index=date, columns=symbol, values=close
        try:
            pivot = df.pivot(index='date', columns='symbol', values='close')
        except Exception:
            return None

        if pivot.empty or len(pivot.columns) < 2:
            return None

        # 计算日收益率
        returns = pivot.pct_change().dropna()

        # 取最近 lookback_window
        if len(returns) > self.lookback_window:
            returns = returns.iloc[-self.lookback_window:]

        if len(returns) < self.min_history:
            return None

        # 只保留有有效数据的列
        returns = returns.dropna(axis=1, how='all')
        available_symbols = [s for s in symbols if s in returns.columns]
        if len(available_symbols) < 2:
            return None

        returns = returns[available_symbols]

        # 填充剩余缺失值（列均值填充）
        returns = returns.fillna(returns.mean())

        # 再次检查是否有足够数据
        if returns.isna().sum().sum() > 0:
            return None

        # 计算协方差矩阵
        cov = returns.cov().values

        # 检查协方差矩阵有效性
        if np.isnan(cov).any() or np.isinf(cov).any():
            return None

        # 正则化：确保正定性
        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals <= 1e-12):
                cov = cov + np.eye(cov.shape[0]) * 1e-6
        except np.linalg.LinAlgError:
            return None

        # 求解风险平价权重
        n = len(available_symbols)
        weights = self._solve_risk_parity(cov)

        if weights is None or len(weights) != n:
            return None

        # 映射回原始 symbols 顺序
        full_weights = np.zeros(len(symbols))
        for i, sym in enumerate(available_symbols):
            idx = symbols.index(sym)
            full_weights[idx] = weights[i]

        # 重新归一化
        if full_weights.sum() > 0:
            full_weights = full_weights / full_weights.sum()
        else:
            return None

        return full_weights

    def _solve_risk_parity(self, cov: np.ndarray,
                           max_iter: int = 100, tol: float = 1e-8) -> Optional[np.ndarray]:
        """使用 Cyclical Coordinate Descent 迭代算法求解风险平价权重。

        迭代公式:
            w_i^{new} = sqrt(w_i * portfolio_var / (n * (Σw)_i))
            然后归一化

        收敛条件:
            ||w_new - w|| < tol
        """
        n = cov.shape[0]

        # 初始化：逆波动率权重
        vol = np.sqrt(np.diag(cov))
        if np.any(vol <= 0):
            return None

        w = 1.0 / vol
        w = w / w.sum()

        for _ in range(max_iter):
            sigma_w = cov @ w
            portfolio_var = w @ sigma_w

            if portfolio_var <= 0 or np.any(sigma_w <= 0):
                return None

            # Cyclical Coordinate Descent 更新
            w_new = np.sqrt(w * portfolio_var / (n * sigma_w))
            w_new = w_new / w_new.sum()

            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break

            w = w_new

        return w

    def _inverse_vol_weights(self, buy_signals) -> np.ndarray:
        """回退方案：逆波动率权重（原简化风险平价）。"""
        volatilities = []
        for sig in buy_signals:
            vol = sig.factors.get(self.volatility_factor, 0.2) if hasattr(sig, 'factors') else 0.2
            vol = max(vol, 0.01)
            volatilities.append(vol)

        inv_vol = np.array([1 / v for v in volatilities])
        weights = inv_vol / inv_vol.sum()
        return weights
