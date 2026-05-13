"""配置管理模块。

存储和加载最佳因子和权重配置。
"""
from __future__ import annotations
import logging
import yaml
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field

from ...database import Database
from .risk_constraints import RiskConstraints


logger = logging.getLogger(__name__)


@dataclass
class FactorConfig:
    """单个因子配置。"""
    name: str
    score: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float = 0.0
    win_rate: float = 0.0


@dataclass
class WeightConfigData:
    """权重配置数据。"""
    name: str
    weights: Dict[str, float]
    score: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float = 0.0
    win_rate: float = 0.0


@dataclass
class StrategyConfig:
    """完整策略配置。"""
    version: str = "1.0"
    created_at: str = ""
    best_factors: List[FactorConfig] = field(default_factory=list)
    best_weight_configs: List[WeightConfigData] = field(default_factory=list)
    risk_constraints: Dict = field(default_factory=dict)


class ConfigManager:
    """配置管理器。"""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        config: StrategyConfig,
        filename: str = "optimized_strategy_config.yaml",
    ):
        """保存配置到YAML文件。"""
        filepath = self.config_dir / filename

        config_dict = {
            "version": config.version,
            "created_at": config.created_at,
            "risk_constraints": config.risk_constraints,
            "best_factors": [
                {
                    "name": f.name,
                    "score": f.score,
                    "annual_return": f.annual_return,
                    "max_drawdown": f.max_drawdown,
                    "sharpe_ratio": f.sharpe_ratio,
                    "calmar_ratio": f.calmar_ratio,
                    "win_rate": f.win_rate,
                }
                for f in config.best_factors
            ],
            "best_weight_configs": [
                {
                    "name": w.name,
                    "weights": w.weights,
                    "score": w.score,
                    "annual_return": w.annual_return,
                    "max_drawdown": w.max_drawdown,
                    "sharpe_ratio": w.sharpe_ratio,
                    "calmar_ratio": w.calmar_ratio,
                    "win_rate": w.win_rate,
                }
                for w in config.best_weight_configs
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"配置已保存到: {filepath}")

    def load(self, filename: str = "optimized_strategy_config.yaml") -> Optional[StrategyConfig]:
        """从YAML文件加载配置。"""
        filepath = self.config_dir / filename

        if not filepath.exists():
            logger.warning(f"配置文件不存在: {filepath}")
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = StrategyConfig(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            risk_constraints=data.get("risk_constraints", {}),
        )

        for f_data in data.get("best_factors", []):
            config.best_factors.append(FactorConfig(
                name=f_data["name"],
                score=f_data["score"],
                annual_return=f_data["annual_return"],
                max_drawdown=f_data["max_drawdown"],
                sharpe_ratio=f_data["sharpe_ratio"],
                calmar_ratio=f_data.get("calmar_ratio", 0.0),
                win_rate=f_data.get("win_rate", 0.0),
            ))

        for w_data in data.get("best_weight_configs", []):
            config.best_weight_configs.append(WeightConfigData(
                name=w_data["name"],
                weights=w_data["weights"],
                score=w_data["score"],
                annual_return=w_data["annual_return"],
                max_drawdown=w_data["max_drawdown"],
                sharpe_ratio=w_data["sharpe_ratio"],
                calmar_ratio=w_data.get("calmar_ratio", 0.0),
                win_rate=w_data.get("win_rate", 0.0),
            ))

        logger.info(f"配置已从 {filepath} 加载")
        return config

    def build_config(
        self,
        factor_results,
        weight_configs,
        risk_constraints: RiskConstraints,
        created_at: str,
    ) -> StrategyConfig:
        """从回测结果构建配置对象。"""
        strategy_config = StrategyConfig(
            version="1.0",
            created_at=created_at,
            risk_constraints=risk_constraints.to_dict(),
        )

        for fr in factor_results:
            calmar = fr.report.annualized_return / abs(fr.report.max_drawdown) if fr.report.max_drawdown != 0 else 0
            strategy_config.best_factors.append(FactorConfig(
                name=fr.factor_name,
                score=fr.score,
                annual_return=fr.report.annualized_return,
                max_drawdown=fr.report.max_drawdown,
                sharpe_ratio=fr.report.sharpe_ratio,
                calmar_ratio=calmar,
                win_rate=fr.report.win_rate,
            ))

        for wc in weight_configs:
            calmar = wc.report.annualized_return / abs(wc.report.max_drawdown) if wc.report.max_drawdown != 0 else 0
            strategy_config.best_weight_configs.append(WeightConfigData(
                name=wc.name,
                weights=wc.weights,
                score=wc.score,
                annual_return=wc.report.annualized_return,
                max_drawdown=wc.report.max_drawdown,
                sharpe_ratio=wc.report.sharpe_ratio,
                calmar_ratio=calmar,
                win_rate=wc.report.win_rate,
            ))

        return strategy_config

    def get_weights(self, config_name: str = "配置-1") -> Optional[Dict[str, float]]:
        """获取特定配置的权重。"""
        config = self.load()
        if config is None:
            return None

        for wc in config.best_weight_configs:
            if wc.name == config_name:
                return wc.weights

        return None

    def get_factor_list(self) -> List[str]:
        """获取因子列表。"""
        config = self.load()
        if config is None:
            return []

        return [f.name for f in config.best_factors]
