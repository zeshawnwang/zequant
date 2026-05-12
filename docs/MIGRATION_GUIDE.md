# 新架构完整迁移总结

## 架构变更概览

本次重构将整个系统彻底迁移到新的信号流架构，完全废弃了旧架构的模块组织方式。

## 核心变更

### 1. 目录结构重整

```
旧架构（已废弃但保留兼容性文件）:
├── screening/
├── timings/
├── portfolios/
└── core/
    ├── strategy.py
    ├── strategy_hub.py
    └── backtest.py

新架构（当前使用）:
core/
├── screening/              # 选股器
│   ├── base/selector.py    # 抽象基类
│   └── impl/               # 具体实现
├── timings/                # 择时器
│   ├── base/timing.py
│   └── impl/
├── positioners/            # 仓位分配器（旧的 portfolios）
│   ├── base/portfolio.py
│   └── impl/
├── signals/                # 信号组合
│   ├── __init__.py         # 组合器 & 约束
│   └── position.py         # 仓位管理器
├── risk/                   # 风控
├── execution/              # 回测引擎
├── monitor/                # 监控
├── research/               # 研究工具
├── optimization/           # 参数优化
└── strategy/               # 策略基类（新）
    ├── __init__.py
    ├── base.py             # SignalStrategy
    └── hub.py              # StrategyHub

兼容性文件（保留但推荐迁移）:
├── core/strategy_legacy.py
└── core/strategy_hub_legacy.py
```

### 2. 新架构核心概念

#### 信号流驱动模型

```
选股得分 (Selector)
    ↓
仓位系数 (PositionSizer)
    ↓
目标权重 (Composer)
    ↓
风控约束 (RiskManager)
    ↓
订单生成 (SignalStrategy)
```

### 3. 模块说明

| 模块 | 职责 | 关键类 |
|------|------|--------|
| **core/screening/** | 股票筛选，输出得分/股票池 | `FactorRankSelector`, `MultiFactorSelector` |
| **core/timings/** | 择时器，输出买卖信号 | `TrendTiming`, `VolatilityTiming` |
| **core/positioners/** | 仓位分配，个股数量/权重 | `EqualWeightBuilder`, `RiskParityBuilder` |
| **core/signals/** | 信号组合，将选股+择时转成目标权重 | `LayeredComposer`, `TrendPositionSizer` |
| **core/risk/** | 风控，约束权重/止损 | `RiskManager`, `StopLoss` |
| **core/execution/** | 执行，回测和实盘 | `BacktestEngine`, `LiveExecutor` |
| **core/strategy/** | 策略组装，积木式组合 | `SignalStrategy`, `CompositeStrategy` |

### 4. 新架构示例策略

```python
from core import (
    SignalStrategy,
    RiskManager,
    BacktestEngine,
)
from core.screening import FactorRankSelector
from core.signals import LayeredComposer, MaxSingleWeightConstraint
from core.signals.position import TrendPositionSizer
from core.strategy import create, list_all, describe

# 方式1：直接构建
strategy = SignalStrategy(
    name="MomentumStrategy_v2",
    selector=FactorRankSelector(factor_name="momentum_20", ascending=False, top_n=90),
    position_sizer=TrendPositionSizer(bullish_threshold=0.6, bearish_threshold=0.4),
    composer=LayeredComposer(
        top_n=30,
        constraints=[MaxSingleWeightConstraint(max_weight=0.1)],
    ),
    risk_manager=RiskManager(max_total_exposure=0.9),
)

# 方式2：通过 StrategyHub（推荐）
import strategies
strategy = strategies.create("momentum_top50", top_n=30)

# 列出所有策略
print("所有策略:", strategies.list_all())
print("策略描述:", strategies.describe("momentum_top50"))
```

## 向后兼容性

虽然新架构是主要推荐的方式，但保留了以下兼容性支持：

1. **QuantStrategy** 可以从 `core` 导入（来自 `strategy_legacy.py`）
2. **旧目录结构的导入** 仍然有效，但推荐迁移
3. **StrategyHub** 支持旧的策略工厂函数

## 迁移指南

### 从旧策略迁移到新策略

1. 继承关系改变：`QuantStrategy` → `SignalStrategy`
2. 组件改变：
   - 择时器 (`timing`) → 仓位管理器 (`position_sizer`)
   - 新增信号组合器 (`composer`)
   - 新增风控管理器 (`risk_manager`)

### 导入路径变化

```python
# 旧导入（不推荐）
from screening import FactorRankSelector
from timings import TrendTiming
from portfolios import EqualWeightBuilder
from core.strategy import QuantStrategy
from core.strategy_hub import register_strategy

# 新导入（推荐）
from core.screening import FactorRankSelector
from core.signals.position import TrendPositionSizer
from core.positioners import EqualWeightBuilder
from core.strategy import SignalStrategy
from core.strategy.hub import register_strategy
```

## 下一步建议

1. **迁移现有策略**：将 `momentum_strategy.py` 等旧策略迁移到新架构
2. **更新测试**：将所有测试更新为使用新架构
3. **添加文档**：为各个核心模块添加使用示例
4. **删除旧文件**：当迁移完成后，删除 `core/strategy_legacy.py` 等兼容文件
