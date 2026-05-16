"""配置驱动的策略工厂。

通过 YAML/JSON 配置自动组装 SignalStrategy 及其所有组件。

用法:
    config = {
        "name": "momentum",
        "top_n": 30,
        "selector": {"type": "factor_rank", "factor_name": "momentum_20", "ascending": False},
        "position_sizer": {"type": "trend", "bullish_threshold": 0.6, "bearish_threshold": 0.4},
        "composer": {
            "type": "layered", "top_n": 30,
            "constraints": [
                {"type": "max_single_weight", "max_weight": 0.1},
            ],
        },
        "risk_manager": {"stop_loss": 0.1, "max_position_pct": 0.1, "max_total_position": 0.9},
    }

    strategy = StrategyFactory.build(config)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from .strategy import SignalStrategy, IStrategy
from ...risk import RiskManager
from ...screening.base.selector_hub import _selector_hub
from ...timings.base.timing_hub import _timing_hub
from ...signals.base.signal_hub import _composer_hub
from ...positioners.base.positioner_hub import _position_sizer_hub
from ...signals.base.composer import (
    MaxSingleWeightConstraint,
    MaxTotalPositionConstraint,
    ReserveCashConstraint,
)

logger = logging.getLogger(__name__)

# 约束类型映射
_CONSTRAINT_MAP = {
    "max_single_weight": MaxSingleWeightConstraint,
    "max_single_position": MaxSingleWeightConstraint,
    "max_total_position": MaxTotalPositionConstraint,
    "reserve_cash": ReserveCashConstraint,
}


class StrategyFactory:
    """配置驱动的策略工厂。"""

    @staticmethod
    def build(config: Dict[str, Any]) -> SignalStrategy:
        """从配置字典构建 SignalStrategy。"""
        name = config.get("name", "Unnamed")
        top_n = config.get("top_n", 30)

        selector = None
        if selector_cfg := config.get("selector"):
            selector = StrategyFactory._build_selector(selector_cfg)

        position_sizer = None
        if sizer_cfg := config.get("position_sizer"):
            position_sizer = StrategyFactory._build_position_sizer(sizer_cfg)

        composer = None
        if composer_cfg := config.get("composer"):
            composer = StrategyFactory._build_composer(composer_cfg)

        risk_manager = None
        if risk_cfg := config.get("risk_manager"):
            risk_manager = StrategyFactory._build_risk_manager(risk_cfg)

        strategy = SignalStrategy(
            name=name,
            selector=selector,
            position_sizer=position_sizer,
            composer=composer,
            risk_manager=risk_manager,
            top_n=top_n,
        )

        logger.info(
            f"配置构建策略: {name} "
            f"(selector={type(selector).__name__ if selector else 'None'}, "
            f"sizer={type(position_sizer).__name__ if position_sizer else 'None'}, "
            f"composer={type(composer).__name__ if composer else 'None'})"
        )
        return strategy

    @staticmethod
    def from_yaml(path: str) -> SignalStrategy:
        """从 YAML 文件加载并构建策略。"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        strategy_cfg = config.get("strategy", config)
        return StrategyFactory.build(strategy_cfg)

    @staticmethod
    def from_json(path: str) -> SignalStrategy:
        """从 JSON 文件加载并构建策略。"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        strategy_cfg = config.get("strategy", config)
        return StrategyFactory.build(strategy_cfg)

    @staticmethod
    def _build_selector(cfg: Dict[str, Any]):
        type_name = cfg.pop("type", "factor_rank")
        return _selector_hub.create(type_name, **cfg)

    @staticmethod
    def _build_position_sizer(cfg: Dict[str, Any]):
        type_name = cfg.pop("type", "trend")
        return _position_sizer_hub.create(type_name, **cfg)

    @staticmethod
    def _build_composer(cfg: Dict[str, Any]):
        type_name = cfg.pop("type", "layered")
        top_n = cfg.pop("top_n", 30)

        constraints_list = cfg.pop("constraints", [])
        constraints = []
        for constraint_cfg in constraints_list:
            constraint_type = constraint_cfg.pop("type", "max_single_weight")
            constraint_cls = _CONSTRAINT_MAP.get(constraint_type)
            if constraint_cls is None:
                logger.warning(f"未知约束类型: {constraint_type}，跳过")
                continue
            constraints.append(constraint_cls(**constraint_cfg))

        if type_name == "layered":
            return _composer_hub.create(type_name, top_n=top_n, constraints=constraints)
        return _composer_hub.create(type_name, constraints=constraints)

    @staticmethod
    def _build_risk_manager(cfg: Dict[str, Any]):
        return RiskManager(config=cfg)
