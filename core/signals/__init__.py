"""信号模块。

包含信号组合器、仓位管理器和对应的注册中心。

目录结构：
  - base/: 基类、组合器、注册中心
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
from .base.signal_hub import (
    ComposerHub,
    register_composer,
    _composer_hub,
)
from .impl.position import (
    IPositionSizer,
    FixedPositionSizer,
    TrendPositionSizer,
    VolatilityPositionSizer,
    RiskParityPositionSizer,
    CompositePositionSizer,
)


def list_composers() -> list:
    """列出所有已注册的组合器。"""
    return _composer_hub.list_all()


__all__ = [
    # 组合器
    "IComposer",
    "IConstraint",
    "LayeredComposer",
    "DirectComposer",
    "WeightedComposer",
    "VoteComposer",
    "MaxSingleWeightConstraint",
    "MaxTotalPositionConstraint",
    "ReserveCashConstraint",
    # 组合器注册中心
    "ComposerHub",
    "register_composer",
    "list_composers",
    # 仓位确定器
    "IPositionSizer",
    "FixedPositionSizer",
    "TrendPositionSizer",
    "VolatilityPositionSizer",
    "RiskParityPositionSizer",
    "CompositePositionSizer",
]
