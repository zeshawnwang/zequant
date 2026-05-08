"""portfolios 模块 —— 仓位分配器(给 BUY 信号分配资金)集合。

公开接口:
  - IPortfolioBuilder:    抽象基类
  - EqualWeightBuilder:    等权重
  - RiskParityBuilder:     风险平价(波动率倒数)
"""
from .base import IPortfolioBuilder
from .equal_weight import EqualWeightBuilder
from .risk_parity import RiskParityBuilder

__all__ = ["IPortfolioBuilder", "EqualWeightBuilder", "RiskParityBuilder"]