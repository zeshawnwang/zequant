# ZeQuant 量化交易系统架构文档

## 系统概述

ZeQuant 是一个专业级量化交易回测系统，采用模块化、信号流驱动的架构设计。

## 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据层 (Data Layer)                      │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ Database    │  │ FactorHub   │  │ UniverseHub │           │
│  │ 数据存储    │  │ 因子注册    │  │ 股票池管理  │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         研究层 (Research Layer)                  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ FactorEval  │  │ Backtest    │  │ Attribution │           │
│  │ 因子评估    │  │ 信号回测    │  │ 绩效归因    │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       信号层 (Signal Layer)                      │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ ISelector   │  │ ITiming     │  │ IComposer   │           │
│  │ 选股信号    │  │ 择时信号    │  │ 信号组合    │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │ IPosition   │  │ IConstraint│                              │
│  │ 仓位管理    │  │ 风控约束    │                              │
│  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       策略层 (Strategy Layer)                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │                   SignalStrategy                       │       │
│  │  - 接收信号流                                         │       │
│  │  - 调用 Composer                                       │       │
│  │  - 输出目标持仓                                        │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       执行层 (Execution Layer)                    │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ Backtester  │  │ Simulator   │  │ LiveTrader │           │
│  │ 历史回测    │  │ 模拟交易    │  │ 实盘对接    │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       监控层 (Monitor Layer)                      │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ Report     │  │ RiskAlert  │  │ FactorMon   │           │
│  │ 绩效报告    │  │ 风险预警    │  │ 因子监控    │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 核心接口设计

### 1. 选股信号接口 (ISelector)

```python
class ISelector(ABC):
    @abstractmethod
    def select(self, date, pool, **kwargs) -> Dict[str, float]:
        """返回股票得分 {symbol: score}"""
        pass
```

### 2. 择时信号接口 (ITiming)

```python
class ITiming(ABC):
    @abstractmethod
    def get_position(self, date, market_data) -> float:
        """返回仓位系数 (0~1)"""
        pass
```

### 3. 信号组合接口 (IComposer)

```python
class IComposer(ABC):
    @abstractmethod
    def compose(
        self,
        selector_scores: Dict[str, float],
        position_signal: float,
        cash: float,
        **kwargs
    ) -> Dict[str, float]:
        """返回目标权重 {symbol: weight}"""
        pass
```

### 4. 仓位管理接口 (IPositionSizer)

```python
class IPositionSizer(ABC):
    @abstractmethod
    def get_position(self, date, market_data) -> float:
        """返回仓位系数 (0~1)"""
        pass
```

### 5. 风控约束接口 (IConstraint)

```python
class IConstraint(ABC):
    @abstractmethod
    def apply(
        self,
        weights: Dict[str, float],
        cash: float,
        **kwargs
    ) -> Dict[str, float]:
        """返回调整后的权重"""
        pass
```

## 策略执行流程

```
1. 获取日期 d 的市场数据
2. 调用 Selector.get_scores(d, market_data) → 选股得分
3. 调用 Timing.get_position(d, market_data) → 仓位系数
4. 调用 Composer.compose(scores, position, cash) → 目标权重
5. 应用所有 Constraints → 最终权重
6. 对比当前持仓和目标权重，生成交易清单
7. 执行交易，更新持仓
8. 记录每日绩效
```

## 模块说明

### 数据层 (Data)

- `Database`: SQLite 数据存储
- `FactorHub`: 因子注册与管理
- `UniverseHub`: 股票池管理

### 信号层 (Signals)

- `ISelector`: 选股信号产生器
- `ITiming`: 择时信号产生器
- `IComposer`: 信号组合器
- `IPositionSizer`: 仓位管理器
- `IConstraint`: 风控约束

### 策略层 (Strategy)

- `SignalStrategy`: 基于信号流的策略
- `Portfolio`: 持仓管理

### 执行层 (Execution)

- `Backtester`: 历史回测引擎
- `Simulator`: 模拟交易
- `LiveTrader`: 实盘交易接口

### 监控层 (Monitor)

- `Report`: 绩效报告
- `RiskAlert`: 风险预警
- `FactorMonitor`: 因子监控

## 使用示例

```python
from core.strategy import SignalStrategy
from core.signals import LayeredComposer, ReserveCashConstraint
from core.signals.position import TrendPositionSizer
from screening import FactorRankSelector
from timings import TrendTiming

# 构建策略
strategy = SignalStrategy(
    selector=FactorRankSelector(factor_name="momentum_20"),
    position_sizer=TrendPositionSizer(),
    composer=LayeredComposer(
        top_n=30,
        constraints=[
            ReserveCashConstraint(reserve_ratio=0.1),
        ],
    ),
)

# 回测
from core.execution import Backtester

backtester = Backtester(initial_capital=1_000_000)
report = backtester.run(strategy, start_date="2020-01-01", end_date="2024-12-31")

print(report.summary())
```

## 扩展指南

### 添加新的选股器

```python
class MySelector(ISelector):
    def select(self, date, pool, **kwargs):
        # 实现选股逻辑
        scores = {}
        for symbol in pool:
            scores[symbol] = self._calc_score(symbol, date)
        return scores
```

### 添加新的择时器

```python
class MyTiming(ITiming):
    def get_position(self, date, market_data):
        # 实现择时逻辑
        return 0.8  # 80% 仓位
```

### 添加新的风控约束

```python
class MyConstraint(IConstraint):
    def apply(self, weights, cash, **kwargs):
        # 实现约束逻辑
        return weights
```

## 文件结构

```
core/
├── __init__.py
├── config.py                 # 配置管理
├── database.py              # 数据存储
├── strategy.py              # 策略基类
├── signals/                 # 信号层
│   ├── __init__.py         # 组合器
│   └── position.py          # 仓位管理
├── execution/               # 执行层
│   ├── __init__.py
│   └── backtester.py
├── monitor/                 # 监控层
│   ├── __init__.py
│   └── report.py
screening/                   # 选股模块
timings/                     # 择时模块
portfolios/                  # 仓位模块
factors/                     # 因子模块
```

## 版本历史

- v1.0: 初始版本
- v2.0: 信号流架构重构（进行中）
