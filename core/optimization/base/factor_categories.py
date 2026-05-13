"""因子分类配置。

将306个因子按10个类别分类，用于分组优化。
"""
from __future__ import annotations
from typing import Dict, List

FACTOR_CATEGORIES: Dict[str, List[str]] = {
    "技术指标": [
        "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29", "a30",
        "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39", "a40",
        "a41", "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49", "a50",
        "a51", "a52", "a53", "a54", "a55", "a56", "a57", "a58", "a59", "a60",
        "a61", "a62", "a63", "a64", "a65", "a66", "a67", "a68", "a69", "a70",
        "a71", "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79", "a80",
        "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89", "a90",
        "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98", "a99", "a100",
        "a101",
        "boll_position", "boll_upper", "boll_middle", "boll_lower",
    ],
    "情绪指标": [
        "rsi_14",
        "macd", "macd_signal", "macd_hist",
    ],
    "动量指标": [
        "momentum_5", "momentum_20",
        "returns",
    ],
    "波动率指标": [
        "volatility_20",
    ],
    "流动性指标": [
        "volume_ratio",
    ],
    "基本面指标": [
        "ff_mkt",
    ],
    "成长性指标": [],
    "质量指标": [],
    "价值指标": [],
    "规模指标": [],
    "其他": [],
}


def get_category_for_factor(factor_name: str) -> str:
    """获取因子所属类别。"""
    for category, factors in FACTOR_CATEGORIES.items():
        if factor_name in factors:
            return category
    return "其他"


def get_factors_by_category(category: str) -> List[str]:
    """获取指定类别的所有因子。"""
    return FACTOR_CATEGORIES.get(category, [])


def get_all_factors() -> List[str]:
    """获取所有已分类的因子。"""
    all_factors = []
    for factors in FACTOR_CATEGORIES.values():
        all_factors.extend(factors)
    return sorted(set(all_factors))


def get_category_summary() -> Dict[str, int]:
    """获取各类别因子数量统计。"""
    return {cat: len(facts) for cat, facts in FACTOR_CATEGORIES.items() if facts}


def get_db_factors_by_category(db) -> Dict[str, List[str]]:
    """从数据库获取因子，并按类别分组。返回数据库中存在的因子。

    将未在预设分类中的因子自动归入"其他"类别。
    """
    db_factors = set(db.list_factor_columns())
    result = {}
    classified_factors = set()

    for category, factors in FACTOR_CATEGORIES.items():
        valid_factors = [f for f in factors if f in db_factors]
        if valid_factors:
            result[category] = valid_factors
            classified_factors.update(valid_factors)

    unclassified = sorted(db_factors - classified_factors)
    if unclassified:
        result["其他"] = unclassified

    return result