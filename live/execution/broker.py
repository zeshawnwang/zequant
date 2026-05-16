"""券商交易接口抽象层。

设计为可插拔:
- PaperBroker:    模拟交易(记录不执行)
- XTPBroker:      中泰XTP
- EastMoneyBroker: 东方财富

新增券商: 继承 BaseBroker 实现 execute_order() 和 query_position() 即可。
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseBroker(ABC):
    @abstractmethod
    def execute_order(self, symbol: str, direction: str, price: float, shares: int) -> dict:
        ...

    @abstractmethod
    def query_position(self, symbol: str) -> dict:
        ...

    @abstractmethod
    def query_account(self) -> dict:
        ...


class PaperBroker(BaseBroker):
    """模拟交易 — 只记录日志，不执行任何实际交易。"""

    def execute_order(self, symbol: str, direction: str, price: float, shares: int) -> dict:
        logger.info("[模拟] %s %s %d股 @ %.2f", direction, symbol, shares, price)
        return {"status": "filled", "symbol": symbol, "direction": direction,
                "price": price, "shares": shares, "trade_id": f"paper_{datetime.now().timestamp()}"}

    def query_position(self, symbol: str) -> dict:
        return {"symbol": symbol, "shares": 0, "cost": 0.0}

    def query_account(self) -> dict:
        return {"total_asset": 0, "cash": 0, "position_value": 0}

    @staticmethod
    def from_config(config: dict) -> "PaperBroker":
        return PaperBroker()


class BrokerFactory:
    """券商工厂 — 根据配置名自动创建。"""
    _registry = {"paper": PaperBroker}

    @classmethod
    def register(cls, name: str, broker_cls):
        cls._registry[name] = broker_cls

    @classmethod
    def create(cls, name: str, config: dict) -> BaseBroker:
        broker_cls = cls._registry.get(name)
        if not broker_cls:
            raise ValueError(f"未知券商: {name}, 可选: {list(cls._registry.keys())}")
        if hasattr(broker_cls, 'from_config'):
            return broker_cls.from_config(config)
        return broker_cls()
