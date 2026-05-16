"""传统技术因子运行器(薄壳)

所有因子都已迁移到 FactorHub 注册体系(见 [`factors/technical.py`](../factors/technical.py) 与
[`factors/alpha101_full.py`](../factors/alpha101_full.py)),本模块仅负责:
  1. 从 DB 拉 K 线
  2. 调 FactorHub.compute_all 计算指定类别的因子
  3. 落库

新增因子只需在 factors/ 下写一个 `@register_factor` 函数即可,无需改本模块。
"""
from __future__ import annotations
import logging
from typing import List, Optional

import pandas as pd

from ...database import Database

logger = logging.getLogger(__name__)


class FactorRunner:
    """因子批量计算 Runner(走 FactorHub 单一注册中心)。"""

    def __init__(self, db: Database):
        self.db = db

    def compute_all(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        names: Optional[List[str]] = None,
        category: str = "technical",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """计算并保存因子。

        Args:
            symbols:    指定股票列表,None 则全市场
            start_date: 入库起始日期(预热数据会先多取再截断,保证 rolling 窗口)
            names:      显式指定要算的因子名;None 时取整个 category
            category:   默认计算 'technical' 类别(13 个传统因子)
            verbose:    打印每个因子的耗时与形状

        Returns:
            pandas.DataFrame: 长表 (date, symbol, factor_name, value),
            同时已写入 DB 的 factors_wide。
        """
        # 懒加载:只在真正调用时才触发因子注册,避免加载 Database 也被迫加载 101 个 alpha
        import core.factors.impl.technical  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        import core.factors.impl.alpha101_full  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        import core.factors.impl.gtja191_full  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        import core.factors.impl.fama_french  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

        bars = self.db.get_daily_bars(start_date=start_date)
        if symbols:
            bars = bars[bars["symbol"].isin(symbols)]

        if bars is None or bars.empty:
            logger.warning("无 K 线数据,请先运行 fetch_data.py")
            return pd.DataFrame()

        if names is None:
            from .factor_hub import list_by_category, compute_all as hub_compute_all
            names = list_by_category(category)
        if not names:
            logger.warning("FactorHub 中无 category=%r 的因子,跳过。", category)
            return pd.DataFrame()

        long_df = hub_compute_all(bars, names=names, verbose=verbose)
        if long_df.empty:
            return long_df

        # 入库前按 start_date 截断(预热段不写)
        if start_date:
            start_ts = pd.Timestamp(start_date).date()
            long_df = long_df[long_df["date"] >= start_ts]

        self.db.save_factors(long_df)
        logger.info(
            "因子计算完成: %d 个因子, %d 条记录",
            long_df["factor_name"].nunique(), len(long_df),
        )
        return long_df