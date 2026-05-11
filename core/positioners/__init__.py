"""仓位分配器模块。

仓位分配器的作用：根据选股和择时信号，决定每个标的的持仓权重。

目录结构:
    core/positioners/
    ├── base/
    │   └── portfolio.py   (基类 IPortfolioBuilder)
    └── impl/
        ├── equal_weight.py (EqualWeightBuilder - 等权重分配)
        └── risk_parity.py (RiskParityBuilder - 风险平价分配)
"""
from .base.portfolio import IPortfolioBuilder
from .impl.equal_weight import EqualWeightBuilder
from .impl.risk_parity import RiskParityBuilder

__all__ = [
    'IPortfolioBuilder',
    'EqualWeightBuilder',
    'RiskParityBuilder',
]
