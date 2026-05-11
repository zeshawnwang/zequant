"""选股器评估引擎 SelectorEvaluator —— 衡量选股器选股质量。

核心指标
--------
- 组合收益率:选股器选出的股票组合的收益率
- 组合超额收益:相对基准（如等权全市场）的超额收益
- 组合最大回撤:组合的最大亏损
- 组合夏普比率:风险调整后收益
- 换手率:选股器的调仓频率
- 胜率:选股组合收益跑赢基准的天数占比
- 盈亏比:超额收益的盈亏比
- 行业分布:组合行业集中度
- 市值分布:组合市值分布
- 因子暴露:组合在各类因子上的暴露

用法
----
    from core.selector_evaluator import SelectorEvaluator
    from screening import TrendBreakoutSelector
    from core.database import Database

    db = Database()
    evaluator = SelectorEvaluator(db)
    selector = TrendBreakoutSelector()
    report = evaluator.evaluate(
        selector=selector,
        start_date="2024-01-01",
        end_date="2024-12-31",
        rebalance_freq="W",
        top_n=50,
    )
    report.pretty_print()
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .database import Database
from .backtest import BacktestReport

logger = logging.getLogger(__name__)


@dataclass
class SelectorEvaluationReport:
    """选股器评估报告数据类。"""
    selector_name: str
    selector_description: str

    # 基础信息
    start_date: str
    end_date: str
    rebalance_freq: str
    top_n: int
    n_rebalances: int

    # 收益指标
    total_return: float = np.nan
    annual_return: float = np.nan
    benchmark_return: float = np.nan
    excess_return: float = np.nan

    # 风险指标
    max_drawdown: float = np.nan
    benchmark_max_drawdown: float = np.nan
    volatility: float = np.nan
    sharpe_ratio: float = np.nan

    # 交易指标
    win_rate: float = np.nan
    profit_loss_ratio: float = np.nan
    turnover_rate: float = np.nan

    # 分布指标
    industry_concentration: Dict[str, float] = field(default_factory=dict)
    market_cap_distribution: Dict[str, float] = field(default_factory=dict)

    def pretty_print(self) -> None:
        """美观打印评估报告。"""
        print("\n" + "=" * 80)
        print(f"选股器评估报告: {self.selector_name}")
        print(f"描述: {self.selector_description}")
        print("=" * 80)

        print(f"\n【基础信息】")
        print(f"  回测期间: {self.start_date} ~ {self.end_date}")
        print(f"  调仓频率: {self.rebalance_freq}")
        print(f"  选股数量: {self.top_n}")
        print(f"  调仓次数: {self.n_rebalances}")

        print(f"\n【收益指标】")
        print(f"  总收益率: {self.total_return:+.2%}")
        print(f"  年化收益: {self.annual_return:+.2%}")
        print(f"  基准收益: {self.benchmark_return:+.2%}")
        print(f"  超额收益: {self.excess_return:+.2%}")

        print(f"\n【风险指标】")
        print(f"  最大回撤: {self.max_drawdown:.2%}")
        print(f"  基准回撤: {self.benchmark_max_drawdown:.2%}")
        print(f"  波动率: {self.volatility:.2%}")
        print(f"  夏普比率: {self.sharpe_ratio:.2f}")

        print(f"\n【交易指标】")
        print(f"  胜率: {self.win_rate:.2%}")
        print(f"  盈亏比: {self.profit_loss_ratio:.2f}")
        print(f"  换手率: {self.turnover_rate:.2%}")

        print(f"\n【行业分布】")
        if self.industry_concentration:
            for industry, weight in sorted(self.industry_concentration.items(), key=lambda x: -x[1])[:5]:
                print(f"  {industry}: {weight:.2%}")

        print(f"\n【市值分布】")
        if self.market_cap_distribution:
            for cap, weight in sorted(self.market_cap_distribution.items(), key=lambda x: -x[1]):
                print(f"  {cap}: {weight:.2%}")
        print("=" * 80 + "\n")

    def to_dict(self) -> Dict:
        """转换为字典格式。"""
        return {
            "selector_name": self.selector_name,
            "selector_description": self.selector_description,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rebalance_freq": self.rebalance_freq,
            "top_n": self.top_n,
            "n_rebalances": self.n_rebalances,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "max_drawdown": self.max_drawdown,
            "benchmark_max_drawdown": self.benchmark_max_drawdown,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "turnover_rate": self.turnover_rate,
            "industry_concentration": self.industry_concentration,
            "market_cap_distribution": self.market_cap_distribution,
        }


class SelectorEvaluator:
    """选股器评估引擎。"""

    def __init__(self, db: Database):
        self.db = db

    def evaluate(
        self,
        selector,
        start_date: str,
        end_date: str,
        rebalance_freq: str = "W",
        top_n: int = 50,
        factor_names: Optional[List[str]] = None,
    ) -> SelectorEvaluationReport:
        """评估单个选股器。

        Args:
            selector: 选股器实例
            start_date: 回测起始日期
            end_date: 回测结束日期
            rebalance_freq: 调仓频率 ("D", "W", "M", "Q")
            top_n: 每次调仓选多少只股票
            factor_names: 需要加载的因子列表（如果为 None 则加载所有）

        Returns:
            SelectorEvaluationReport: 评估报告
        """
        from .screening.base.selector import IStockSelector
        if not isinstance(selector, IStockSelector):
            raise TypeError(f"selector must implement IStockSelector, got {type(selector)}")

        logger.info(
            "评估选股器: %s, 期间=%s~%s, 调仓=%s, top_n=%d",
            selector.get_description(), start_date, end_date, rebalance_freq, top_n,
        )

        # 1. 加载因子数据
        if factor_names is None:
            factor_names = self.db.list_factor_columns()

        factor_data = self.db.get_factors(
            factor_names=factor_names,
            start_date=start_date,
            end_date=end_date,
            with_close=True,
        )
        if factor_data is None or factor_data.empty:
            return self._empty_report(
                selector, start_date, end_date, rebalance_freq, top_n
            )

        factor_data["date"] = pd.to_datetime(factor_data["date"])
        factor_data = factor_data.sort_values(["date", "symbol"])

        # 2. 加载日K数据
        bars = self.db.get_daily_bars(
            start_date=start_date,
            end_date=end_date,
            columns=["date", "symbol", "close"],
        )
        if bars is None or bars.empty:
            return self._empty_report(
                selector, start_date, end_date, rebalance_freq, top_n
            )

        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.sort_values(["date", "symbol"])

        # 3. 生成调仓日期
        rebalance_dates = self._generate_rebalance_dates(
            bars["date"].min(),
            bars["date"].max(),
            rebalance_freq,
        )

        # 4. 模拟选股和持仓
        portfolio_returns, benchmark_returns, turnover = self._simulate_portfolio(
            selector=selector,
            factor_data=factor_data,
            bars=bars,
            rebalance_dates=rebalance_dates,
            top_n=top_n,
        )

        # 5. 计算评估指标
        return self._compute_report(
            selector=selector,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            turnover=turnover,
            rebalance_dates=rebalance_dates,
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            top_n=top_n,
        )

    def compare_selectors(
        self,
        selectors: List,
        start_date: str,
        end_date: str,
        rebalance_freq: str = "W",
        top_n: int = 50,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """批量对比多个选股器。

        Args:
            selectors: 选股器列表
            start_date: 回测起始日期
            end_date: 回测结束日期
            rebalance_freq: 调仓频率
            top_n: 每次调仓选多少只股票
            factor_names: 需要加载的因子列表

        Returns:
            pd.DataFrame: 对比表格
        """
        reports = []
        for selector in selectors:
            try:
                report = self.evaluate(
                    selector=selector,
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_freq=rebalance_freq,
                    top_n=top_n,
                    factor_names=factor_names,
                )
                reports.append(report)
            except Exception as e:
                logger.warning("评估 %s 失败: %s", selector.get_description(), e)

        if not reports:
            return pd.DataFrame()

        rows = [r.to_dict() for r in reports]
        df = pd.DataFrame(rows)

        # 按超额收益排序
        df = df.sort_values("excess_return", ascending=False)

        return df

    # ---- 内部辅助方法 --------------------------------------------------------

    def _empty_report(
        self,
        selector,
        start_date: str,
        end_date: str,
        rebalance_freq: str,
        top_n: int,
    ) -> SelectorEvaluationReport:
        """生成空报告。"""
        return SelectorEvaluationReport(
            selector_name=type(selector).__name__,
            selector_description=selector.get_description(),
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            top_n=top_n,
            n_rebalances=0,
        )

    def _generate_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        freq: str,
    ) -> List[pd.Timestamp]:
        """生成调仓日期序列。"""
        freq_map = {
            "D": "D",
            "W": "W",
            "M": "M",
            "Q": "Q",
        }
        dates = pd.date_range(start=start_date, end=end_date, freq=freq_map.get(freq, "W"))
        return list(dates)

    def _simulate_portfolio(
        self,
        selector,
        factor_data: pd.DataFrame,
        bars: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        top_n: int,
    ) -> Tuple[pd.Series, pd.Series, float]:
        """模拟组合运行，返回(组合收益序列, 基准收益序列, 换手率)。"""
        # 准备价格数据
        price_pivot = bars.pivot(index="date", columns="symbol", values="close")

        # 准备因子数据
        factor_data_pivot = factor_data.set_index(["date", "symbol"])

        # 模拟持仓
        current_holdings = set()
        portfolio_values = []
        turnover_list = []
        dates_in_range = []

        valid_rebalance_dates = []
        for dt in rebalance_dates:
            if dt in factor_data["date"].values:
                valid_rebalance_dates.append(dt)

        if not valid_rebalance_dates:
            return pd.Series(), pd.Series(), 0.0

        # 从第一个调仓日开始
        start_dt = valid_rebalance_dates[0]
        current_value = 1.0
        portfolio_values.append((start_dt, current_value))

        for i, dt in enumerate(valid_rebalance_dates):
            dates_in_range.append(dt)

            # 选股
            try:
                date_factor_data = factor_data[factor_data["date"] == dt].copy()
                selected = selector.select(date_factor_data, top_n=top_n)
            except Exception as e:
                logger.warning("在 %s 选股失败: %s", dt, e)
                selected = []

            # 计算换手率
            if current_holdings:
                overlap = len(current_holdings & set(selected))
                turn = 1 - overlap / len(current_holdings) if current_holdings else 0.0
                turnover_list.append(turn)

            # 更新持仓
            current_holdings = set(selected)

            # 计算到下一个调仓日的收益
            next_dt = valid_rebalance_dates[i + 1] if i + 1 < len(valid_rebalance_dates) else None
            if next_dt is None:
                break

            period_prices = price_pivot.loc[dt:next_dt]
            if len(period_prices) < 2:
                continue

            # 计算等权组合收益
            holdings_list = list(current_holdings)
            if not holdings_list:
                continue

            available = [s for s in holdings_list if s in period_prices.columns]
            if not available:
                continue

            # 等权
            initial_prices = period_prices.iloc[0][available]
            final_prices = period_prices.iloc[-1][available]

            period_returns = (final_prices / initial_prices - 1).mean()
            current_value *= (1 + period_returns)
            portfolio_values.append((next_dt, current_value))

        # 构建收益序列
        if len(portfolio_values) < 2:
            return pd.Series(), pd.Series(), 0.0

        portfolio_series = pd.Series(
            [v for _, v in portfolio_values],
            index=[d for d, _ in portfolio_values],
            name="portfolio"
        )
        portfolio_returns = portfolio_series.pct_change().dropna()

        # 构建基准（等权全市场）
        benchmark_series = price_pivot.mean(axis=1).loc[portfolio_series.index]
        benchmark_returns = benchmark_series.pct_change().dropna()

        # 平均换手率
        avg_turnover = np.mean(turnover_list) if turnover_list else 0.0

        return portfolio_returns, benchmark_returns, avg_turnover

    def _compute_report(
        self,
        selector,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        turnover: float,
        rebalance_dates: List[pd.Timestamp],
        start_date: str,
        end_date: str,
        rebalance_freq: str,
        top_n: int,
    ) -> SelectorEvaluationReport:
        """计算评估报告指标。"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return self._empty_report(selector, start_date, end_date, rebalance_freq, top_n)

        # 计算组合累计收益
        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()

        total_return = portfolio_cum.iloc[-1] - 1 if len(portfolio_cum) > 0 else 0.0
        benchmark_return = benchmark_cum.iloc[-1] - 1 if len(benchmark_cum) > 0 else 0.0
        excess_return = total_return - benchmark_return

        # 计算年化收益
        n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        annual_return = (1 + total_return) ** (365 / max(n_days, 1)) - 1 if n_days > 0 else 0.0

        # 计算最大回撤
        portfolio_maxdd = self._compute_max_drawdown(portfolio_cum)
        benchmark_maxdd = self._compute_max_drawdown(benchmark_cum)

        # 计算波动率和夏普比率
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0

        # 计算胜率和盈亏比
        excess_rets = portfolio_returns - benchmark_returns
        win_rate = (excess_rets > 0).mean()

        positive_rets = excess_rets[excess_rets > 0]
        negative_rets = excess_rets[excess_rets <= 0]
        profit_loss_ratio = (
            positive_rets.mean() / abs(negative_rets.mean())
            if len(positive_rets) > 0 and len(negative_rets) > 0 and negative_rets.mean() != 0
            else 1.0
        )

        return SelectorEvaluationReport(
            selector_name=type(selector).__name__,
            selector_description=selector.get_description(),
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            top_n=top_n,
            n_rebalances=len(rebalance_dates),
            total_return=float(total_return),
            annual_return=float(annual_return),
            benchmark_return=float(benchmark_return),
            excess_return=float(excess_return),
            max_drawdown=float(portfolio_maxdd),
            benchmark_max_drawdown=float(benchmark_maxdd),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            win_rate=float(win_rate),
            profit_loss_ratio=float(profit_loss_ratio),
            turnover_rate=float(turnover),
        )

    @staticmethod
    def _compute_max_drawdown(cum_returns: pd.Series) -> float:
        """计算最大回撤。"""
        if cum_returns.empty or len(cum_returns) < 2:
            return 0.0

        cummax = cum_returns.cummax()
        drawdown = (cum_returns - cummax) / cummax
        return float(drawdown.min())
