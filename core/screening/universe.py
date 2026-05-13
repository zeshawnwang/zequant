"""Universe 过滤器 —— 对全市场股票做"可交易性"过滤,产生当日候选池。

过滤维度:
  1. ST/*ST/退市:不交易
  2. 上市不满 N 天:不交易(新股波动剧烈)
  3. 当日涨跌停:不【买入】(已持仓的卖出不受限)
     - 板块分级判定(2020 年后新规):
       * 创业板(30开头) / 科创板(688开头):±20%
       * 北交所(4/8 开头,排除 ST):±30%
       * ST / *ST:±5%
       * 其他(沪深主板):±10%
  4. 停牌:当日无 K 线视为停牌,不交易
  5. 流动性不足:amount 低于阈值,不买入

接口:
  - SymbolUniverse 一次性预加载 symbols 元信息
  - filter_buyable(date, bars_today) -> Set[str]:可买入集合
  - is_sellable(symbol, bars_today) -> bool:是否可卖出
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Set, Optional
import pandas as pd

from ..database import Database

logger = logging.getLogger(__name__)


# ===== 涨跌停幅度判定 =====================================================

def get_price_limit_pct(symbol: str, is_st: bool = False) -> float:
    """按板块/股票类型返回涨跌停幅度(百分比,如 10.0 表示 ±10%)。

    Args:
        symbol: 股票代码,6 位数字字符串
        is_st:  是否 ST/*ST 股

    Returns:
        允许的最大单日涨跌幅(正数,单位 %)
    """
    if is_st:
        return 5.0  # ST/*ST:±5%
    s = str(symbol).zfill(6)
    # 科创板:688xxx、689xxx
    if s.startswith(("688", "689")):
        return 20.0
    # 创业板:300xxx、301xxx
    if s.startswith(("300", "301")):
        return 20.0
    # 北交所:4xxxxx、83xxxx、87xxxx、88xxxx、92xxxx 等
    if s.startswith(("43", "83", "87", "88", "92")) or s.startswith("4"):
        return 30.0
    # 默认沪深主板
    return 10.0


# ===== Universe 配置 =====================================================

@dataclass
class UniverseConfig:
    """Universe 过滤配置(对应 config.yaml 的 universe 段)。"""
    exclude_st: bool = True
    min_listed_days: int = 60
    exclude_limit_up: bool = True       # 涨停日不买入
    exclude_limit_down: bool = True     # 跌停日不卖出(也无法成交)
    min_daily_amount: float = 0.0       # 当日成交额下限(元)
    limit_buffer: float = 0.2           # 涨跌停判定缓冲区(百分点),避免恰好踩线的噪声

    @classmethod
    def from_config(cls, cfg: dict) -> "UniverseConfig":
        """从 config.yaml 的 universe 子段构造。

        兼容旧配置:exclude 列表里可能是字符串 'ST股' 或 dict {'上市不满N天': 60}。
        """
        if not cfg:
            return cls()
        exclude_st = True
        min_listed = 60
        for item in cfg.get("exclude", []) or []:
            if isinstance(item, str) and "ST" in item.upper():
                exclude_st = True
            elif isinstance(item, dict):
                for k, v in item.items():
                    if "上市" in k or "list" in k.lower():
                        min_listed = int(v)
        return cls(
            exclude_st=exclude_st,
            min_listed_days=min_listed,
            min_daily_amount=float(cfg.get("min_daily_amount", 0) or 0),
        )

    def __repr__(self) -> str:
        return (
            f"UniverseConfig(exclude_st={self.exclude_st}, "
            f"min_listed_days={self.min_listed_days}, "
            f"min_daily_amount={self.min_daily_amount:,.0f}元, "
            f"板块分级涨跌停=是)"
        )


# ===== Universe 主类 =====================================================

class SymbolUniverse:
    """股票池过滤器。"""

    def __init__(self, db: Database, config: Optional[UniverseConfig] = None):
        self.db = db
        self.config = config or UniverseConfig()
        self._symbols_df: Optional[pd.DataFrame] = None
        self._st_symbols: Set[str] = set()
        self._load()

    def _load(self) -> None:
        """加载 symbols 元信息,预计算 ST 集合。"""
        self._symbols_df = self.db.get_symbols()
        if self._symbols_df is None or self._symbols_df.empty:
            self._symbols_df = pd.DataFrame(columns=["symbol", "name", "list_date"])
            return
        if self.config.exclude_st and "name" in self._symbols_df.columns:
            mask = (
                self._symbols_df["name"]
                .fillna("")
                .str.upper()
                .str.contains("ST", na=False)
            )
            self._st_symbols = set(self._symbols_df.loc[mask, "symbol"])
        if "list_date" in self._symbols_df.columns:
            self._symbols_df["list_date"] = pd.to_datetime(
                self._symbols_df["list_date"], errors="coerce"
            )

    # ---- 辅助:涨跌停判定 -------------------------------------------------

    def _limit_mask(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """向量化的涨/跌停判定,返回布尔 Series(True 表示踩线)。

        Args:
            df: 必须含 symbol, pct_change 两列(pct_change 以百分比为单位,
                例如 9.5 表示 +9.5%)
            direction: 'up' 或 'down'
        """
        buf = float(self.config.limit_buffer)
        # 为每行计算其所在板块的涨跌停阈值
        is_st = df["symbol"].isin(self._st_symbols)
        thresh = df["symbol"].apply(
            lambda s: get_price_limit_pct(s, is_st=False)
        )
        # ST 股在上面 apply 时没带 is_st 标志,在这里单独覆盖
        thresh = thresh.where(~is_st, 5.0)

        pct = pd.to_numeric(df["pct_change"], errors="coerce").astype(float)
        if direction == "up":
            return pct >= (thresh - buf)
        else:
            return pct <= -(thresh - buf)

    # ---- 主入口 -----------------------------------------------------------

    def filter_buyable(
        self,
        date,
        daily_bars_today: pd.DataFrame,
        candidate_symbols: Optional[Set[str]] = None,
    ) -> Set[str]:
        """给定 date 与当日 daily_bars,返回该日【可买入】的 symbol 集合。

        Args:
            date:              当日日期
            daily_bars_today:  当日日线快照(date/symbol/pct_change/amount/volume)
            candidate_symbols: 可选候选池(缩小扫描范围,用于性能优化)
        """
        date = pd.Timestamp(date)
        if daily_bars_today is None or daily_bars_today.empty:
            return set()

        df = daily_bars_today.copy()
        if candidate_symbols is not None:
            df = df[df["symbol"].isin(candidate_symbols)]
        if df.empty:
            return set()

        # 1) ST 股
        if self.config.exclude_st and self._st_symbols:
            df = df[~df["symbol"].isin(self._st_symbols)]

        # 2) 上市不满 N 天
        if self.config.min_listed_days > 0 and not self._symbols_df.empty:
            cutoff = date - pd.Timedelta(days=self.config.min_listed_days)
            ok_symbols = self._symbols_df.loc[
                (self._symbols_df["list_date"].notna())
                & (self._symbols_df["list_date"] <= cutoff),
                "symbol",
            ]
            unknown = self._symbols_df.loc[
                self._symbols_df["list_date"].isna(), "symbol"
            ]
            allow = set(ok_symbols) | set(unknown)
            df = df[df["symbol"].isin(allow)]

        # 3) 涨停日不买入(板块分级)
        if self.config.exclude_limit_up and "pct_change" in df.columns and not df.empty:
            mask = self._limit_mask(df, "up")
            df = df[~mask]

        # 4) 流动性
        if self.config.min_daily_amount > 0 and "amount" in df.columns:
            df = df[df["amount"].astype(float) >= self.config.min_daily_amount]

        # 5) 停牌(volume=0)
        if "volume" in df.columns:
            df = df[df["volume"].astype(float) > 0]

        return set(df["symbol"])

    def is_sellable(self, symbol: str, daily_bars_today: pd.DataFrame) -> bool:
        """给定 symbol 在当日是否可【卖出】。

        - 当日无 bars(停牌)→ 不可卖
        - 跌停 & exclude_limit_down → 不可卖(也无法成交)
        """
        if daily_bars_today is None or daily_bars_today.empty:
            return False
        row = daily_bars_today[daily_bars_today["symbol"] == symbol]
        if row.empty:
            return False
        if "volume" in row.columns and float(row["volume"].iloc[0]) <= 0:
            return False
        if (
            self.config.exclude_limit_down
            and "pct_change" in row.columns
        ):
            pc = float(row["pct_change"].iloc[0])
            is_st = symbol in self._st_symbols
            thresh = get_price_limit_pct(symbol, is_st=is_st)
            if pc <= -(thresh - self.config.limit_buffer):
                return False
        return True