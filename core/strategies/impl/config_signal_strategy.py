"""新架构策略配置示例

展示如何通过配置文件定制策略积木。
"""
import logging
from typing import Dict, Any

# =============================================================================
# 策略配置示例
# =============================================================================

STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 配置 1: 激进型动量策略
    "momentum_aggressive": {
        "strategy": "momentum_v2",
        "top_n": 20,
        "selector": {
            "factor_name": "momentum_20",
            "ascending": False,
        },
        "position_sizer": {
            "sma_short": 3,
            "sma_medium": 10,
            "buy_threshold": 0.5,
            "sell_threshold": 0.3,
        },
        "risk": {
            "max_single_position": 0.10,
            "max_total_exposure": 0.95,
            "stop_loss_threshold": 0.15,
        },
    },
    # 配置 2: 稳健型低波动策略
    "low_vol_defensive": {
        "strategy": "low_vol_v2",
        "top_n": 40,
        "selector": {
            "factor_name": "volatility_20",
            "ascending": True,
        },
        "position_sizer": {
            "volatility_factor": "volatility_20",
            "target_volatility": 0.15,
        },
        "risk": {
            "max_single_position": 0.05,
            "max_total_exposure": 0.80,
            "stop_loss_threshold": 0.08,
        },
    },
    # 配置 3: 平衡型复合策略
    "trend_vol_balanced": {
        "strategy": "trend_vol_v2",
        "top_n": 30,
        "selector": {
            "factor_name": "momentum_20",
            "ascending": False,
        },
        "position_sizer": {
            "sma_short": 5,
            "sma_medium": 20,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "high_threshold": 0.05,
            "low_threshold": 0.03,
        },
        "risk": {
            "max_single_position": 0.05,
            "max_total_exposure": 0.80,
            "stop_loss_threshold": 0.10,
            "take_profit_threshold": 0.25,
        },
    },
}


def get_config(name: str) -> Dict[str, Any]:
    """获取策略配置"""
    return STRATEGY_CONFIGS.get(name, {})


def list_configs() -> list:
    """列出所有配置"""
    return list(STRATEGY_CONFIGS.keys())


if __name__ == "__main__":
    print("新架构策略配置")
    print("=" * 60)
    
    print("\n可用配置:")
    for name in list_configs():
        print(f"\n  [{name}]")
        cfg = get_config(name)
        print(f"    策略类型: {cfg['strategy']}")
        print(f"    持仓数量: {cfg['top_n']}")
        print(f"    选股因子: {cfg['selector']['factor_name']}")
