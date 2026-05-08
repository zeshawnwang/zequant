"""项目级配置加载器(零依赖,纯函数)。

统一从 [`config/config.yaml`](../config/config.yaml) 读出全部段;调用方按需取字段,
避免每个 script 各自 PyYAML 解析散落,便于将来切换配置后端(env / TOML / DB)。

用法
----
    from core.config import load_config
    cfg = load_config()                       # 默认 config/config.yaml
    cfg = load_config("path/to/other.yaml")
    db_path        = cfg["database"]["path"]
    ir_threshold   = cfg["factors"]["ir_threshold"]
    forward_days   = cfg["factors"]["forward_days"]

约定
----
- 找不到字段时返回 schema 内置的默认值,而非抛异常,保证旧配置文件平滑升级
- 路径相对工作目录,不做 cwd 推断;调用方负责 chdir 或传绝对路径
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# 默认配置 —— 与 config/config.yaml 内容一致;字段缺失时用此兜底
DEFAULT_CONFIG: Dict[str, Any] = {
    "database": {"path": "./data/quant_data.db"},
    "data_source": {"primary": "akshare", "fallback": "tushare"},
    "universe": {
        "exclude": ["ST股", {"上市不满N天": 60}],
        "min_daily_amount": 100_000_000,
    },
    "fees": {
        "stamp_tax": 0.001,
        "transfer_fee": 0.00002,
        "commission": 0.0003,
        "min_commission": 5,
        "slippage": 0.0005,
    },
    "risk": {
        "max_position_pct": 0.15,
        "max_total_position": 0.85,
        "stop_loss": 0.10,
        "take_profit": 0.25,
    },
    "backtest": {
        "initial_capital": 1_000_000,
        "rebalance_freq": "1d",
        "start_date": "2019-01-01",
        "end_date": "2026-05-01",
    },
    "factors": {
        "ir_threshold": 0.05,
        "forward_days": 5,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """把 override 深合并到 base(返回新 dict),仅 dict 递归,其他类型直接覆盖。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = "config/config.yaml") -> Dict[str, Any]:
    """加载并返回合并了默认值的配置 dict。

    Args:
        path: yaml 路径;None 或文件不存在时直接返回 DEFAULT_CONFIG 副本

    Returns:
        dict —— DEFAULT_CONFIG 与文件内容深合并的结果
    """
    if not path:
        return _deep_merge(DEFAULT_CONFIG, {})
    p = Path(path)
    if not p.exists():
        return _deep_merge(DEFAULT_CONFIG, {})
    with open(p, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def get_db_path(cfg: Dict[str, Any]) -> str:
    """便捷读取 database.path,带兜底。"""
    return str(cfg.get("database", {}).get("path") or DEFAULT_CONFIG["database"]["path"])