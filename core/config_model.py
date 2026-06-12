"""
Pydantic 配置校验模型。

所有配置段对应 config/config.yaml，在 load_config() 加载后过此 schema，
确保缺失值、类型错误、费率新旧版本等问题在启动时（而非运行时）暴露。
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Dict 向下兼容 Mixin
# ============================================================

class _DictCompatBase(BaseModel):
    """为所有子模型添加 cfg['key'] 风格访问，避免改现有调用方。"""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


# ============================================================
# 数据库
# ============================================================

class DatabaseConfig(_DictCompatBase):
    path: str = "./data/quant_data.db"
    threads: int | None = None  # None = auto-detect (os.cpu_count())
    memory_limit: str | None = None  # None = auto-detect


# ============================================================
# 数据源
# ============================================================

class DataSourceConfig(_DictCompatBase):
    primary: str = "akshare"
    fallback: str = "baostock"


# ============================================================
# Universe 过滤
# ============================================================

class UniverseExcludeConfig(_DictCompatBase):
    st: bool = Field(default=True, alias="ST股")
    min_days_since_listing: int = Field(default=60, alias="上市不满N天")

    @model_validator(mode="before")
    @classmethod
    def unpack_list(cls, values: Any) -> Any:
        """YAML 中 exclude 是混合列表 ['ST股', {'上市不满N天': 60}]，
        转成结构化字段。"""
        if isinstance(values, list):
            result: dict = {}
            for item in values:
                if isinstance(item, str) and "ST" in item.upper():
                    result["ST股"] = True
                elif isinstance(item, dict):
                    result.update(item)
            return result
        if isinstance(values, dict):
            return values
        return {}


class UniverseConfig(_DictCompatBase):
    exclude: UniverseExcludeConfig = Field(default_factory=UniverseExcludeConfig)
    min_daily_amount: float = 100_000_000


# ============================================================
# 交易费用 (A 股实盘标准 2025)
# ============================================================

class FeesConfig(_DictCompatBase):
    stamp_tax: float = 0.0005      # 印花税 0.05%，卖出单边（2023.8.28 起）
    transfer_fee: float = 0.00001  # 过户费 0.001%，沪深统一（2022 年起）
    commission: float = 0.0003     # 佣金万三
    min_commission: float = 5.0    # 最低佣金 5 元
    slippage: float = 0.0005       # 滑点 0.05%

    @field_validator("stamp_tax")
    @classmethod
    def warn_old_tax(cls, v: float) -> float:
        if v >= 0.001:
            import warnings
            warnings.warn(
                f"印花税 {v} 可能是旧税率（2023.8.28 起应为 0.0005），请确认 config"
            )
        return v


# ============================================================
# 风控
# ============================================================

class RiskConfig(_DictCompatBase):
    max_position_pct: float = 0.15
    max_total_position: float = 0.85
    stop_loss: float = 0.10
    take_profit: float = 0.25


# ============================================================
# 回测
# ============================================================

class BacktestConfig(_DictCompatBase):
    initial_capital: float = 1_000_000
    rebalance_freq: str = "1d"
    start_date: str = "2019-01-01"
    end_date: str = "2026-05-01"


# ============================================================
# 因子
# ============================================================

class FactorsConfig(_DictCompatBase):
    ir_threshold: float = 0.05
    forward_days: int = 5


# ============================================================
# 顶级 Config
# ============================================================

class ZeQuantConfig(_DictCompatBase):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    factors: FactorsConfig = Field(default_factory=FactorsConfig)
    strategies: Dict[str, Any] = Field(default_factory=dict)

    # ---- 构造 ----

    @classmethod
    def from_dict(cls, data: dict) -> "ZeQuantConfig":
        """从 YAML 解析后的 dict 构造，只取顶层级已知段，
        未知段（如 strategies 内部参数）保留为 dict。"""
        cleaned = {}
        known_sections = {"database", "data_source", "universe",
                          "fees", "risk", "backtest", "factors", "strategies"}
        for key in data:
            if key in known_sections:
                cleaned[key] = data[key]
        return cls(**cleaned)
