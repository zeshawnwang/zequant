"""风控约束模块。

定义策略的风控约束条件，并提供检查功能。
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """风控检查结果。"""
    passed: bool
    violated_constraints: List[str]
    details: Dict

    @property
    def message(self) -> str:
        if self.passed:
            return "通过所有风控检查"
        return "以下风控约束被违反: " + "; ".join(self.violated_constraints)


class RiskConstraints:
    """策略风控约束集合。"""

    def __init__(
        self,
        max_drawdown: float = 0.20,
        single_stock_weight: float = 0.15,
        single_sector_weight: float = 0.25,
        max_volatility: float = 0.30,
        max_turnover: float = 1.00,
        min_calmar_ratio: float = 0.5,
        min_win_rate: float = 0.50,
    ):
        self.max_drawdown = max_drawdown
        self.single_stock_weight = single_stock_weight
        self.single_sector_weight = single_sector_weight
        self.max_volatility = max_volatility
        self.max_turnover = max_turnover
        self.min_calmar_ratio = min_calmar_ratio
        self.min_win_rate = min_win_rate

    def check_backtest_result(
        self,
        annual_return: float,
        max_drawdown: float,
        volatility: float,
        calmar_ratio: float,
        win_rate: float,
        turnover: Optional[float] = None,
    ) -> RiskCheckResult:
        """检查回测结果是否满足风控要求。"""
        violated: List[str] = []
        details: Dict = {}

        if max_drawdown > self.max_drawdown:
            violated.append(f"最大回撤 {max_drawdown:.2%} > {self.max_drawdown:.2%}")
            details["max_drawdown"] = max_drawdown

        if volatility > self.max_volatility:
            violated.append(f"波动率 {volatility:.2%} > {self.max_volatility:.2%}")
            details["volatility"] = volatility

        if calmar_ratio < self.min_calmar_ratio:
            violated.append(f"Calmar 比率 {calmar_ratio:.2f} < {self.min_calmar_ratio:.2f}")
            details["calmar_ratio"] = calmar_ratio

        if win_rate < self.min_win_rate:
            violated.append(f"胜率 {win_rate:.2%} < {self.min_win_rate:.2%}")
            details["win_rate"] = win_rate

        if turnover is not None and turnover > self.max_turnover:
            violated.append(f"换手率 {turnover:.2%} > {self.max_turnover:.2%}")
            details["turnover"] = turnover

        return RiskCheckResult(
            passed=len(violated) == 0,
            violated_constraints=violated,
            details=details,
        )

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            "max_drawdown": self.max_drawdown,
            "single_stock_weight": self.single_stock_weight,
            "single_sector_weight": self.single_sector_weight,
            "max_volatility": self.max_volatility,
            "max_turnover": self.max_turnover,
            "min_calmar_ratio": self.min_calmar_ratio,
            "min_win_rate": self.min_win_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RiskConstraints":
        """从字典创建。"""
        return cls(**data)
