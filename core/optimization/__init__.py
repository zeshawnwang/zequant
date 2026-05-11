"""策略优化模块。

包含因子筛选、权重优化、策略归因、配置管理等模块。
"""
from .risk_constraints import RiskConstraints, RiskCheckResult
from .factor_selector import FactorSelector
from .weight_optimizer import GeneticWeightOptimizer
from .attribution import StrategyAttribution
from .config_manager import StrategyConfig, ConfigManager

__all__ = [
    'RiskConstraints',
    'RiskCheckResult',
    'FactorSelector',
    'GeneticWeightOptimizer',
    'StrategyAttribution',
    'StrategyConfig',
    'ConfigManager',
]
