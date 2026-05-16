"""仓位分配器模块。

仓位分配器的作用：根据选股和择时信号，决定每个标的的持仓权重。

目录结构:
    core/positioners/
    ├── base/
    │   ├── portfolio.py      (基类 IPortfolioBuilder)
    │   └── positioner_hub.py (PositionSizerHub 注册中心)
    └── impl/
        ├── equal_weight.py   (EqualWeightBuilder - 等权重分配)
        ├── risk_parity.py    (RiskParityBuilder - 风险平价分配，完整IPortfolioBuilder)
        └── rp_weights.py     (RPPortfolioWeights - 纯numpy底层风险平价分配)
"""
from .base.portfolio import IPortfolioBuilder
from .base.positioner_hub import PositionSizerHub, register_position_sizer, _position_sizer_hub
from .impl.equal_weight import EqualWeightBuilder
from .impl.risk_parity import RiskParityBuilder
from .impl.rp_weights import RPPortfolioWeights


def list_position_sizers() -> list:
    """列出所有已注册的仓位确定器。"""
    return _position_sizer_hub.list_all()


__all__ = [
    'IPortfolioBuilder',
    'PositionSizerHub',
    'register_position_sizer',
    'list_position_sizers',
    'EqualWeightBuilder',
    'RiskParityBuilder',
    'RPPortfolioWeights',
]
