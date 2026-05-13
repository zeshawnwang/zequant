"""项目级配置加载器(零依赖,纯函数)。

统一从 [`config/config.yaml`](../config/config.yaml) 读出全部段;调用方按需取字段,
避免每个 script 各自 PyYAML 解析散落,便于将来切换配置后端(env / TOML / DB)。

注意：这里不再维护重复的默认配置了！唯一真实配置源是 config/config.yaml。

用法
----
    from core.config import load_config
    cfg = load_config()                       # 默认 config/config.yaml
    cfg = load_config("path/to/other.yaml")
    db_path        = cfg["database"]["path"]
    ir_threshold   = cfg["factors"]["ir_threshold"]
    forward_days   = cfg["factors"]["forward_days"]

    # 读取策略专属配置
    strat_cfg = get_strategy_config(cfg, "momentum_top50")
    top_n = strat_cfg.get("top_n", 50)

约定
----
- 找不到配置文件时会抛出异常，不再隐藏
- 路径相对工作目录，不做 cwd 推断；调用方负责 chdir 或传绝对路径
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """从 YAML 文件加载完整配置。

    Args:
        config_path: 配置文件路径，默认 "config/config.yaml"

    Returns:
        完整配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 解析错误
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_strategy_config(cfg: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """从完整配置中提取策略专属配置段。

    Args:
        cfg: load_config() 返回的完整配置
        strategy_name: 策略名（对应 cfg["strategies"] 中的键）

    Returns:
        策略专属配置字典，没有时返回空字典
    """
    return cfg.get("strategies", {}).get(strategy_name, {})
