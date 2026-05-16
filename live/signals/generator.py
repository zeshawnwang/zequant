"""策略信号生成 — 调用core/strategies/pipeline.py 生成今日持仓建议。"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self, strategy_name: str = None):
        self.strategy_name = strategy_name or "mf_vol_d10_rp"

    def generate(self) -> list[dict]:
        """
        生成今日调仓信号。

        返回:
            [{"symbol":"000001","direction":"buy","weight":0.05,"reason":"MF信号排名前10"}, ...]
        """
        # TODO: 接入完整的PyTorch/numpy推理管线
        # 当前为 MVP 占位，后续接入 Pipeline.build_signal()
        logger.info("信号生成: strategy=%s", self.strategy_name)
        logger.warning("MVP: 使用占位逻辑，需接入 Pipeline")
        return []
