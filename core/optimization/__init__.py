"""策略优化模块。

包含因子筛选、权重优化、策略归因、向量化评估等模块。

目录结构：
  - base/: 基类和配置管理
  - impl/: 具体实现
"""
from .base.config_manager import StrategyConfig, ConfigManager
from .base.risk_constraints import RiskConstraints, RiskCheckResult
from .impl.vectorized_evaluator import VectorizedEvaluator, EvalResult
from .impl.weight_optimizer import GeneticWeightOptimizer, WeightConfig
from .impl.attribution import StrategyAttribution, AttributionResult

__all__ = [
    "StrategyConfig",
    "ConfigManager",
    "RiskConstraints",
    "RiskCheckResult",
    "VectorizedEvaluator",
    "EvalResult",
    "GeneticWeightOptimizer",
    "WeightConfig",
    "StrategyAttribution",
    "AttributionResult",
]
