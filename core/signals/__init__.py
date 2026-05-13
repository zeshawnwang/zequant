"""信号模块。

包含信号组合器和仓位管理器。

目录结构：
  - base/: 基类和组合器
  - impl/: 具体仓位管理器实现
"""
from .base.composer import (
    IComposer,
    IConstraint,
    LayeredComposer,
    DirectComposer,
    WeightedComposer,
    VoteComposer,
    MaxSingleWeightConstraint,
    MaxTotalPositionConstraint,
    ReserveCashConstraint,
)
from .impl.position import TrendPositionSizer, VolatilityPositionSizer

__all__ = [
    "IComposer",
    "IConstraint",
    "LayeredComposer",
    "DirectComposer",
    "WeightedComposer",
    "VoteComposer",
    "MaxSingleWeightConstraint",
    "MaxTotalPositionConstraint",
    "ReserveCashConstraint",
    "TrendPositionSizer",
    "VolatilityPositionSizer",
]
