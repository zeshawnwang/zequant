"""因子库入口。

因子分类：
  - base/: 基类和因子注册中心
  - impl/: 具体因子实现
"""
from .base.factor_hub import FactorHub

__all__ = ["FactorHub"]
