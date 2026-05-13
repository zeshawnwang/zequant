"""性能监控模块

功能：
- 计算组合绩效指标（收益、回撤、夏普、卡玛等）
- 分层分析（选股、择时、风控贡献）
- 归因分析（Brinson模型）
- 风险指标（波动率、VaR、CVaR）
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DrawdownInfo:
    """回撤信息"""
    max_drawdown: float
    max_drawdown_date: str
    recovery_date: Optional[str]
    recovery_days: Optional[int]
    current_drawdown: float
    current_drawdown_date: str


@dataclass
class RiskMetrics:
    """风险指标"""
    volatility: float
    downside_volatility: float
    var_95: float
    cvar_95: float
    max_loss: float
    skewness: float
    kurtosis: float


@dataclass
class PeriodReturn:
    """分段收益"""
    period: str
    start_date: str
    end_date: str
    return_val: float
    annualized: float


@dataclass
class PerformanceReport:
    """绩效报告"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_date: str = ""
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    period_returns: List[PeriodReturn] = field(default_factory=list)
    monthly_returns: pd.DataFrame = None
    yearly_returns: pd.DataFrame = None
    drawdown_info: DrawdownInfo = None
    risk_metrics: RiskMetrics = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'total_return': f"{self.total_return*100:.2f}%",
            'annualized_return': f"{self.annualized_return*100:.2f}%",
            'volatility': f"{self.volatility*100:.2f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'calmar_ratio': f"{self.calmar_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown*100:.2f}%",
            'win_rate': f"{self.win_rate*100:.2f}%",
            'profit_factor': f"{self.profit_factor:.2f}",
            'total_trades': self.total_trades,
        }


class PerformanceMonitor:
    """性能监控器

    计算和分析组合绩效指标
    """

    RISK_FREE_RATE = 0.03

    def __init__(self, risk_free_rate: float = None):
        self.risk_free_rate = risk_free_rate or self.RISK_FREE_RATE

    def analyze(
        self,
        equity_curve: pd.DataFrame,
        trades: List[dict] = None,
        benchmark: pd.Series = None,
    ) -> PerformanceReport:
        """分析绩效

        Args:
            equity_curve: 净值曲线，包含date和total_value列
            trades: 交易记录列表
            benchmark: 基准收益序列
        """
        if equity_curve is None or equity_curve.empty:
            return PerformanceReport()

        df = equity_curve.copy()
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df.index)
        df = df.sort_values('date')

        total_value = df['total_value'].astype(float)
        returns = total_value.pct_change().dropna()

        report = PerformanceReport()

        report.total_return = self._calc_total_return(total_value)
        report.annualized_return = self._calc_annualized_return(total_value, df)
        report.volatility = self._calc_volatility(returns)
        report.sharpe_ratio = self._calc_sharpe(returns)
        report.sortino_ratio = self._calc_sortino(returns)
        report.max_drawdown, report.max_drawdown_date, report.max_drawdown_duration = self._calc_max_drawdown(df)
        report.calmar_ratio = abs(report.max_drawdown) / report.annualized_return if report.annualized_return > 0 else 0

        if trades:
            report.total_trades = len(trades)
            report.win_rate, report.profit_factor = self._calc_trade_stats(trades)

        report.period_returns = self._calc_period_returns(df)
        report.monthly_returns = self._calc_monthly_returns(df)
        report.yearly_returns = self._calc_yearly_returns(df)

        dd_info = self._calc_drawdown_info(df)
        report.drawdown_info = dd_info

        risk_metrics = self._calc_risk_metrics(returns)
        report.risk_metrics = risk_metrics

        return report

    def _calc_total_return(self, total_value: pd.Series) -> float:
        """计算总收益"""
        if len(total_value) < 2:
            return 0.0
        return float((total_value.iloc[-1] / total_value.iloc[0]) - 1)

    def _calc_annualized_return(self, total_value: pd.Series, df: pd.DataFrame) -> float:
        """计算年化收益"""
        if len(total_value) < 2:
            return 0.0
        initial = float(total_value.iloc[0])
        final = float(total_value.iloc[-1])
        if initial <= 0 or final <= 0:
            return 0.0

        start_date = pd.to_datetime(df['date'].iloc[0])
        end_date = pd.to_datetime(df['date'].iloc[-1])
        years = (end_date - start_date).days / 365.25
        if years <= 0:
            return 0.0

        return float((final / initial) ** (1 / years) - 1)

    def _calc_volatility(self, returns: pd.Series) -> float:
        """计算波动率"""
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    def _calc_sharpe(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess_return = returns.mean() * 252 - self.risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return float(excess_return / volatility)

    def _calc_sortino(self, returns: pd.Series) -> float:
        """计算索提诺比率"""
        if len(returns) < 2:
            return 0.0
        excess_return = returns.mean() * 252 - self.risk_free_rate
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_vol = downside_returns.std() * np.sqrt(252)
        if downside_vol == 0:
            return 0.0
        return float(excess_return / downside_vol)

    def _calc_max_drawdown(
        self,
        df: pd.DataFrame,
    ) -> Tuple[float, str, int]:
        """计算最大回撤"""
        total_value = df['total_value'].astype(float)
        cummax = total_value.cummax()
        drawdown = (total_value - cummax) / cummax

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        max_dd_date = str(pd.Timestamp(df.loc[max_dd_idx, 'date']).date()) if max_dd_idx in df.index else ""

        peak_idx = total_value[:max_dd_idx].idxmax()
        recovery_idx = total_value[max_dd_idx:][total_value[max_dd_idx:] >= total_value[peak_idx]].index
        duration = 0
        if len(recovery_idx) > 0:
            recovery_date = recovery_idx[0]
            duration = int((recovery_date - max_dd_idx).days)
        else:
            duration = int((df['date'].iloc[-1] - max_dd_idx).days)

        return float(max_dd), max_dd_date, duration

    def _calc_trade_stats(self, trades: List[dict]) -> Tuple[float, float]:
        """计算交易统计"""
        if not trades:
            return 0.0, 0.0

        realized = []
        for i, trade in enumerate(trades):
            if trade.get('direction') == 'SELL':
                pnl = trade.get('pnl', 0)
                if pnl:
                    realized.append(pnl)

        if not realized:
            return 0.0, 0.0

        wins = [p for p in realized if p > 0]
        losses = [p for p in realized if p < 0]
        win_rate = len(wins) / len(realized)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        return float(win_rate), float(profit_factor)

    def _calc_period_returns(self, df: pd.DataFrame) -> List[PeriodReturn]:
        """计算分段收益"""
        periods = []
        initial = float(df['total_value'].iloc[0])

        period_configs = [
            ('1M', 21),
            ('3M', 63),
            ('6M', 126),
            ('YTD', None),
        ]

        dates = pd.to_datetime(df['date'])
        current_year = dates.max().year

        for name, days in period_configs:
            if name == 'YTD':
                ytd_df = df[dates.dt.year == current_year]
                if len(ytd_df) > 1:
                    start_val = float(ytd_df['total_value'].iloc[0])
                    end_val = float(ytd_df['total_value'].iloc[-1])
                    ret = (end_val / start_val) - 1
                    start_date = str(pd.Timestamp(ytd_df['date'].iloc[0]).date())
                    end_date = str(pd.Timestamp(ytd_df['date'].iloc[-1]).date())
                    ann = ret * (252 / len(ytd_df)) if len(ytd_df) > 0 else 0
                    periods.append(PeriodReturn(name, start_date, end_date, float(ret), float(ann)))
            elif days and len(df) >= days:
                start_val = float(df['total_value'].iloc[-days])
                end_val = float(df['total_value'].iloc[-1])
                ret = (end_val / start_val) - 1
                start_date = str(pd.Timestamp(df['date'].iloc[-days]).date())
                end_date = str(pd.Timestamp(df['date'].iloc[-1]).date())
                ann = ret * (252 / days)
                periods.append(PeriodReturn(name, start_date, end_date, float(ret), float(ann)))

        return periods

    def _calc_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算月度收益"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        monthly = df['total_value'].resample('M').last().pct_change()
        monthly.index = monthly.index.strftime('%Y-%m')
        monthly = monthly.dropna()
        return monthly

    def _calc_yearly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算年度收益"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        yearly = df['total_value'].resample('Y').last().pct_change()
        yearly.index = yearly.index.year
        yearly = yearly.dropna()
        return yearly

    def _calc_drawdown_info(self, df: pd.DataFrame) -> DrawdownInfo:
        """计算回撤信息"""
        total_value = df['total_value'].astype(float)
        cummax = total_value.cummax()
        drawdown = (total_value - cummax) / cummax

        max_dd = float(drawdown.min())
        max_dd_idx = drawdown.idxmin()
        max_dd_date = str(pd.Timestamp(df.loc[max_dd_idx, 'date']).date()) if max_dd_idx in df.index else ""

        current_dd = float(drawdown.iloc[-1])
        current_dd_date = str(pd.Timestamp(df['date'].iloc[-1]).date())

        peak_idx = total_value[:max_dd_idx].idxmax()
        recovery_idx = total_value[max_dd_idx:][total_value[max_dd_idx:] >= total_value[peak_idx]].index
        recovery_date = None
        recovery_days = None
        if len(recovery_idx) > 0:
            recovery_date = str(pd.Timestamp(recovery_idx[0]).date())
            recovery_days = int((recovery_idx[0] - max_dd_idx).days)

        return DrawdownInfo(
            max_drawdown=max_dd,
            max_drawdown_date=max_dd_date,
            recovery_date=recovery_date,
            recovery_days=recovery_days,
            current_drawdown=current_dd,
            current_drawdown_date=current_dd_date,
        )

    def _calc_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """计算风险指标"""
        if len(returns) < 2:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

        volatility = float(returns.std() * np.sqrt(252))
        downside_returns = returns[returns < 0]
        downside_vol = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0.0

        sorted_returns = returns.sort_values()
        var_idx = int(len(sorted_returns) * 0.05)
        var_95 = float(sorted_returns.iloc[var_idx]) if var_idx < len(sorted_returns) else 0.0
        cvar_95 = float(sorted_returns.iloc[:var_idx].mean()) if var_idx > 0 else 0.0

        max_loss = float(returns.min())
        skewness = float(returns.skew())
        kurtosis = float(returns.kurtosis())

        return RiskMetrics(
            volatility=volatility,
            downside_volatility=downside_vol,
            var_95=var_95,
            cvar_95=cvar_95,
            max_loss=max_loss,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def compare_with_benchmark(
        self,
        equity_curve: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> Dict:
        """与基准对比"""
        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        portfolio_returns = df['total_value'].pct_change().dropna()

        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) == 0:
            return {}

        portfolio_returns = portfolio_returns[common_idx]
        benchmark_returns = benchmark_returns[common_idx]

        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = float(excess_returns.std() * np.sqrt(252))
        information_ratio = float(excess_returns.mean() * 252 / tracking_error) if tracking_error > 0 else 0.0

        beta = float(portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()) if benchmark_returns.var() > 0 else 1.0
        alpha = float((portfolio_returns.mean() - beta * benchmark_returns.mean()) * 252)

        correlation = float(portfolio_returns.corr(benchmark_returns))

        return {
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'excess_return': float(excess_returns.mean() * 252),
        }
