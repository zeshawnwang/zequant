# 新架构积木式策略构建指南

## 🧱 概述

新架构采用**信号流驱动**的设计，允许你像搭积木一样灵活组合不同的模块来构建策略。

---

## 🎛️ 模块选择指南

### 第1层：选股器（Selector）
选择用什么逻辑挑选股票

| 选股器 | 说明 | 适用场景 |
|--------|------|----------|
| `FactorRankSelector` | 按单因子排名选股 | 简单的动量、低波、估值等策略 |
| `MultiFactorSelector` | 多因子综合选股 | 复杂的多因子策略 |
| `CustomSelector` | 自定义选股逻辑 | 特殊业务逻辑 |

**示例**：
```python
# 动量选股（选 momentum_20 最高的）
selector = FactorRankSelector(
    factor_name="momentum_20",
    ascending=False,
    top_n=90,
)
```

---

### 第2层：仓位管理器（PositionSizer）
选择如何控制总仓位

| 仓位管理器 | 说明 | 仓位系数 |
|------------|------|----------|
| `TrendPositionSizer` | 趋势择时 | 0~1 |
| `VolatilityPositionSizer` | 波动率择时 | 0~1 |
| `FixedPositionSizer` | 固定仓位 | 固定值 |

**示例**：
```python
# 趋势择时
position_sizer = TrendPositionSizer(
    sma_short=5,
    sma_medium=20,
    buy_threshold=0.6,
    sell_threshold=0.4,
)
```

---

### 第3层：信号组合器（Composer）
选择如何组合选股和择时信号

| 组合器 | 说明 | 推荐场景 |
|--------|------|----------|
| `LayeredComposer` | 分层组合（先择时再分配） | 最常用的直观方式 |
| `DirectComposer` | 直接相乘 | 简单的数学组合 |
| `VoteComposer` | 投票组合 | 多信号融合 |
| `WeightedComposer` | 加权组合 | 需要调整权重 |

**示例**：
```python
# 分层组合
composer = LayeredComposer(
    top_n=30,
    constraints=[
        MaxSingleWeightConstraint(max_weight=0.05),
        ReserveCashConstraint(reserve_ratio=0.05),
    ],
)
```

---

### 第4层：风控管理器（RiskManager）
选择风控规则

| 风控组件 | 说明 |
|----------|------|
| `MaxSingleWeightConstraint` | 单票权重上限 |
| `MaxTotalPositionConstraint` | 总仓位上限 |
| `MinPositionConstraint` | 最小仓位约束 |
| `ReserveCashConstraint` | 预留现金 |
| `TurnoverConstraint` | 换手率约束 |
| `StopLoss` | 止损规则（固定/移动） |
| `TakeProfit` | 止盈规则 |

**示例**：
```python
# 风控管理
risk_manager = RiskManager(
    constraints=[
        MaxTotalPositionConstraint(max_total=0.95),
        MinPositionConstraint(min_single=0.005),
    ],
    stop_loss=StopLoss(method="fixed", threshold=0.10),
    take_profit=TakeProfit(method="fixed", threshold=0.30),
    max_total_exposure=0.95,
    max_single_position=0.05,
)
```

---

## 🛠️ 完整示例

### 示例1：激进型动量策略

```python
from core.strategy.base import SignalStrategy
from core.signals import LayeredComposer, MaxSingleWeightConstraint, ReserveCashConstraint
from core.signals.position import TrendPositionSizer
from core.risk import RiskManager, StopLoss, TakeProfit
from screening.factor_rank import FactorRankSelector

strategy = SignalStrategy(
    name="AggressiveMomentum",
    
    # 选股器：动量
    selector=FactorRankSelector(
        factor_name="momentum_20",
        ascending=False,
        top_n=60,
    ),
    
    # 仓位：快速趋势
    position_sizer=TrendPositionSizer(
        sma_short=3,
        sma_medium=10,
        buy_threshold=0.5,
        sell_threshold=0.3,
    ),
    
    # 组合：重仓集中
    composer=LayeredComposer(
        top_n=20,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.10),
            ReserveCashConstraint(reserve_ratio=0.05),
        ],
    ),
    
    # 风控：高风险容忍
    risk_manager=RiskManager(
        stop_loss=StopLoss(method="fixed", threshold=0.15),
        max_total_exposure=0.95,
        max_single_position=0.10,
    ),
)
```

### 示例2：保守型低波动策略

```python
strategy = SignalStrategy(
    name="ConservativeLowVol",
    
    # 选股：低波动
    selector=FactorRankSelector(
        factor_name="volatility_20",
        ascending=True,
        top_n=120,
    ),
    
    # 仓位：波动率控制
    position_sizer=VolatilityPositionSizer(
        target_volatility=0.15,
        max_position=0.8,
    ),
    
    # 组合：分散配置
    composer=LayeredComposer(
        top_n=40,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.05),
            ReserveCashConstraint(reserve_ratio=0.20),
        ],
    ),
    
    # 风控：严格
    risk_manager=RiskManager(
        stop_loss=StopLoss(method="fixed", threshold=0.08),
        max_total_exposure=0.80,
        max_single_position=0.05,
    ),
)
```

---

## 🚀 快速开始

### 1. 查看示例策略

```bash
cd /Users/wangzeshang1/MyProjects/zequant
python3 strategies/example_signal_strategy.py
```

### 2. 使用预配置策略

```python
from strategies.example_signal_strategy import STRATEGIES
from strategies.config_signal_strategy import get_config

# 使用预配置
config = get_config("low_vol_defensive")
strategy = STRATEGIES[config["strategy"]](
    top_n=config["top_n"],
    strategy_config=config,
)
```

### 3. 运行回测

```bash
python3 scripts/example_backtest_signal.py
```

---

## 📊 配置选项

查看 `strategies/config_signal_strategy.py` 了解完整的预配置选项：

- `momentum_aggressive`: 激进型动量
- `low_vol_defensive`: 保守型低波动
- `trend_vol_balanced`: 平衡型复合

---

## 💡 进阶技巧

### 自定义组合器

继承 `IComposer` 类：
```python
class MyCustomComposer(IComposer):
    def compose(self, selector_scores, position_signal, cash, current_weights, **kwargs):
        # 你的组合逻辑
        pass
```

### 自定义约束

继承 `IConstraint` 类：
```python
class MyCustomConstraint(IConstraint):
    def apply(self, weights, cash, positions):
        # 你的约束逻辑
        pass
```

---

## 🎯 架构优势

1. **模块化**：每个部分独立，可单独测试
2. **灵活**：随意搭配不同组件
3. **清晰**：执行流程一目了然
4. **可扩展**：容易添加新的组件类型

---

## 📞 支持

如有问题，请参考：
- `core/strategy/base.py` - 策略基类定义
- `core/signals/__init__.py` - 信号组合器
- `core/risk/__init__.py` - 风控管理器
- `strategies/example_signal_strategy.py` - 完整示例
