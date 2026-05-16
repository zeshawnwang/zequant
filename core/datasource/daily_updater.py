"""
日频数据更新管道 — orchestrator脚本的核心调度模块。

职责：
  1. 增量拉取日线数据（IncrementalFetcher）
  2. 增量计算注册因子（FactorRunner — technical/alpha101/gtja）
  3. 增量计算自定义技术因子（ma5/ma20/ma_convergence等）
  4. 数据质量检查
  5. 写入 update_log

用法：
    from core.datasource.daily_updater import DailyUpdater
    updater = DailyUpdater()
    updater.run()

路径约定：所有数据操作走 core/datasource/，scripts/ 只做 CLI 入口。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from ..database import Database

logger = logging.getLogger(__name__)

# 自定义技术因子（依赖ma5/ma20等，不在FactorHub注册体系中）
TECHNICAL_FACTORS = [
    'ma5', 'ma10', 'ma20', 'ma21', 'ma60', 'ma120',
    'ma_alignment_score', 'ma60_trend', 'ma120_trend',
    'macd_above_zero', 'macd_golden_cross',
    'volume_breakout_ratio', 'volume_contraction',
    'ma_convergence', 'chip_concentration', 'ma_angle_20',
]


class DailyUpdater:
    """统一日频数据更新调度器。"""

    def __init__(self, db: Optional[Database] = None, start_date: Optional[str] = None):
        self.db = db or Database()
        self.start_date = start_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    def run(self, fetch_bars: bool = False, compute_factors: bool = True,
            compute_technical: bool = True, check_quality: bool = True):
        """全量运行更新管道。

        Args:
            fetch_bars:   是否拉取日线（默认False，因akshare逐只拉取较慢）
            compute_factors: 是否计算FactorHub注册因子
            compute_technical: 是否计算自定义技术因子
            check_quality: 是否进行数据质量检查
        """
        logger.info("=" * 60)
        logger.info("DailyUpdater 开始运行")
        logger.info("=" * 60)

        if fetch_bars:
            self._fetch_daily_bars()
        if compute_factors:
            self._compute_factors()
        if compute_technical:
            self._compute_technical_factors(force=True)
        if check_quality:
            self._check_quality()

        self._log_update()
        logger.info("DailyUpdater 完成")

    def _fetch_daily_bars(self):
        """拉取日线数据（FallbackFetcher 兜底链）。"""
        from .fallback_fetcher import FallbackFetcher
        fetcher = FallbackFetcher()
        logger.info("拉取日线数据 [兜底链=%s]...", "+".join(fetcher.priority))

        # 获取活跃股票
        try:
            sym_df = self.db.conn.execute("""
                SELECT symbol FROM daily_bars
                GROUP BY symbol HAVING COUNT(*) >= 60
            """).df()
            symbols = sym_df["symbol"].tolist()
        except Exception:
            sym_df = self.db.get_symbols()
            symbols = sym_df["symbol"].tolist() if not sym_df.empty else []

        if not symbols:
            logger.warning("无活跃股票")
            return

        logger.info("  %d 只活跃股票需检查", len(symbols))
        count = 0
        for i, sym in enumerate(symbols):
            if (i + 1) % 200 == 0:
                logger.info("  [%d/%d] %s ...", i + 1, len(symbols), sym)
            df = fetcher.fetch_bars(sym, self.start_date)
            if len(df) > 0:
                self.db.upsert_daily_bars(df)
                count += 1

        logger.info("日线拉取完成: %d/%d 只更新", count, len(symbols))

    def _compute_factors(self):
        """计算FactorHub注册因子(technical/alpha101等)。"""
        from ..factors.base.factor import FactorRunner
        runner = FactorRunner(self.db)
        logger.info("计算注册因子...")

        for category in ['technical']:
            logger.info("  计算 category=%s ...", category)
            df = runner.compute_all(start_date=self.start_date, category=category)
            if not df.empty:
                logger.info("  ✅ %s: %d 条", category, len(df))

    def _compute_technical_factors(self, force: bool = False):
        """计算自定义技术因子（ma5/ma20等，不在FactorHub中）。

        Args:
            force: 为True时强制重新计算最新日期的数据
        """
        logger.info("计算自定义技术因子...")

        existing = set(self.db.list_factor_columns())
        needed = [c for c in TECHNICAL_FACTORS if c not in existing]

        if force:
            logger.info("  强制重算全部 %d 个技术因子 (最新区间 %s)", len(TECHNICAL_FACTORS), self.start_date)
            needed = TECHNICAL_FACTORS
        elif not needed:
            logger.info("  所有技术因子数据已是最新，跳过")
            return
        else:
            logger.info("  新增 %d 个技术因子列: %s", len(needed), needed[:5])

        # 加载日线
        bars = self.db.get_daily_bars(
            columns=['symbol', 'date', 'close', 'volume'],
            start_date=self.start_date,
        )
        if bars.empty:
            logger.warning("  无日线数据，跳过")
            return

        bars['date'] = pd.to_datetime(bars['date'])
        bars = bars.sort_values(['symbol', 'date']).reset_index(drop=True)

        # 计算SMA
        logger.info("  计算SMA因子: %s", needed)
        for w in [5, 10, 20, 21, 60, 120]:
            col = f'ma{w}'
            if col in needed:
                bars[col] = bars.groupby('symbol')['close'].transform(
                    lambda x: x.rolling(window=w, min_periods=max(3, w // 3)).mean()
                )

        # 衍生因子
        if 'ma_alignment_score' in needed:
            bars['ma_alignment_score'] = (
                (bars.get('ma5', 0) > bars.get('ma20', 0)).astype(float) +
                (bars.get('ma20', 0) > bars.get('ma60', 0)).astype(float) +
                (bars.get('ma60', 0) > bars.get('ma120', 0)).astype(float)
            ) / 3.0

        if 'ma60_trend' in needed:
            bars['ma60_trend'] = bars.groupby('symbol')['ma60'].transform(
                lambda x: x / x.shift(5) - 1.0
            )
        if 'ma120_trend' in needed:
            bars['ma120_trend'] = bars.groupby('symbol')['ma120'].transform(
                lambda x: x / x.shift(10) - 1.0
            )

        # 加载已有因子用于衍生计算
        existing_factors = self.db.get_factors(
            factor_names=['macd', 'macd_signal', 'volume', 'volatility_20'],
            start_date=self.start_date,
        )
        if existing_factors is not None and not existing_factors.empty:
            existing_factors['date'] = pd.to_datetime(existing_factors['date'])
            bars = bars.merge(
                existing_factors[['date', 'symbol', 'macd', 'macd_signal']],
                on=['date', 'symbol'], how='left'
            )
            if 'macd_above_zero' in needed:
                bars['macd_above_zero'] = (bars.get('macd', 0) > 0).astype(float)
            if 'macd_golden_cross' in needed:
                bars['macd_golden_cross'] = (
                    (bars.get('macd', 0) > bars.get('macd_signal', 0)) &
                    (bars.groupby('symbol')['macd'].shift(1) <= bars.groupby('symbol')['macd_signal'].shift(1))
                ).astype(float)

        # 量比
        if 'volume_breakout_ratio' in needed:
            bars['volume_ma20'] = bars.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(20).mean()
            )
            bars['volume_breakout_ratio'] = np.where(
                bars['volume_ma20'] > 0,
                bars['volume'] / (bars['volume_ma20'] + 1), 1.0
            )
        if 'volume_contraction' in needed:
            bars['volume_ma5'] = bars.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(5).mean()
            )
            bars['volume_contraction'] = np.where(
                bars['volume_ma20'] > 0,
                bars['volume_ma5'] / (bars['volume_ma20'] + 1), 1.0
            )

        # 均线粘合度
        if 'ma_convergence' in needed:
            ma_cols = [c for c in ['ma5', 'ma20', 'ma60'] if c in bars]
            if len(ma_cols) >= 2:
                bars['ma_mean'] = bars[ma_cols].mean(axis=1)
                bars['ma_std'] = bars[ma_cols].std(axis=1)
                bars['ma_convergence'] = np.where(
                    bars['ma_mean'] > 0,
                    bars['ma_std'] / (bars['ma_mean'] + 1e-10), 1.0
                )

        # 筹码集中度
        if 'chip_concentration' in needed:
            bars['price_vol'] = bars.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20).std()
            )
            bars['chip_concentration'] = np.where(
                bars['price_vol'] > 0,
                1.0 / (1.0 + bars['price_vol'] / bars['close']), 0.0
            )

        # 均线角度
        if 'ma_angle_20' in needed:
            bars['ma_angle_20'] = bars.groupby('symbol')['ma20'].transform(
                lambda x: x / x.shift(5) - 1.0
            )

        # 写入数据库
        save_cols = ['date', 'symbol'] + needed
        save_cols = [c for c in save_cols if c in bars.columns]
        save_df = bars[save_cols].copy()
        for c in needed:
            if c in save_df.columns:
                save_df[c] = save_df[c].replace([np.inf, -np.inf], np.nan)

        save_df = save_df.dropna(subset=needed, how='all')
        if not save_df.empty:
            self.db.ensure_factor_columns(needed)
            self.db.save_factors(save_df)
            logger.info("  ✅ 技术因子写入完成: %d 条", len(save_df))

    def _check_quality(self):
        """数据质量检查。"""
        from .checker import DataQualityChecker
        checker = DataQualityChecker(self.db)
        issues = checker.check_all(start=self.start_date)
        if issues:
            logger.warning("数据质量: %d 个问题", len(issues))
            for issue in issues[:5]:
                logger.warning("  %s", issue)
        else:
            logger.info("数据质量: ✅ 通过")

    def _log_update(self):
        """记录更新日志。"""
        try:
            self.db.conn.execute("""
                INSERT INTO update_log (table_name, last_update, records_updated, status)
                VALUES ('daily_updater', CURRENT_TIMESTAMP, 1, 'SUCCESS')
            """)
        except Exception:
            pass
