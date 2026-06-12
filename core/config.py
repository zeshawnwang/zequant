"""项目级配置加载器（Pydantic 校验版）。

统一从 config/config.yaml 读出全部段，过 Pydantic schema 校验后返回。
调用方通过字段访问（cfg.database.path / cfg.fees.stamp_tax），
替代手写 dict 键名拼写。

用法
----
    from core.config import load_config
    cfg = load_config()                       # ZeQuantConfig 对象
    db_path      = cfg.database.path
    ir_threshold = cfg.factors.ir_threshold
    forward_days = cfg.factors.forward_days

    # 读取策略专属配置（仍为 dict，因为是动态结构）
    strat_cfg = cfg.strategies.get("momentum_top50", {})
    top_n = strat_cfg.get("top_n", 50)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

from core.config_model import ZeQuantConfig


def load_config(config_path: str | Path = "config/config.yaml") -> ZeQuantConfig:
    """从 YAML 加载并校验配置。

    Args:
        config_path: 配置文件路径，默认 "config/config.yaml"

    Returns:
        经过 Pydantic 校验的 ZeQuantConfig 对象

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 解析错误
        pydantic.ValidationError: 配置字段类型/值校验失败
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    return ZeQuantConfig.from_dict(raw)


def get_strategy_config(cfg: ZeQuantConfig, strategy_name: str) -> Dict[str, Any]:
    """从 ZeQuantConfig 中提取策略专属配置段。

    Args:
        cfg: load_config() 返回的 ZeQuantConfig 对象
        strategy_name: 策略名（对应 cfg.strategies 中的键）

    Returns:
        策略配置 dict，没有时返回空 dict
    """
    return cfg.strategies.get(strategy_name, {})
