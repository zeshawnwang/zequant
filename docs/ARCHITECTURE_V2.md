# ZeQuant 架构重构完成总结

## 架构概览

所有模块已统一整理到 `core/` 目录下，基类与实现分离：

```
/Users/wangzeshang1/MyProjects/zequant/
├── core/                                    # 核心模块
│   ├── screening/                           # 选股器
│   │   ├── base/
│   │   │   └── selector.py                 (基类 IStockSelector)
│   │   └── impl/
│   │       ├── factor_rank.py              (FactorRankSelector)
│   │       ├── multi_factor.py             (MultiFactorSelector)
│   │       ├── fundamental.py              (FundamentalSelector)
│   │       └── momentum_breakout.py        (TrendBreakoutSelector, OversoldReboundSelector, ChipConcentrationSelector)
│   │
│   ├── timings/                             # 择时器
│   │   ├── base/
│   │   │   └── timing.py                   (基类 ITimingGenerator)
│   │   └── impl/
│   │       ├── trend.py                    (TrendTiming)
│   │       ├── volatility.py               (VolatilityTiming)
│   │       ├── trend_volatility.py         (TrendVolatilityTiming)
│   │       ├── combo.py                    (ComboTiming)
│   │       └── market_regime.py            (MarketRegimeTiming)
│   │
│   ├── positioners/                         # 仓位分配器 (原 portfolios)
│   │   ├── base/
│   │   │   └── portfolio.py                (基类 IPortfolioBuilder)
│   │   └── impl/
│   │       ├── equal_weight.py             (EqualWeightBuilder)
│   │       └── risk_parity.py              (RiskParityBuilder)
│   │
│   ├── signals/                             # 信号组合器 (新架构 SignalStrategy 使用)
│   │   ├── __init__.py                     (IComposer, LayeredComposer, DirectComposer, etc.)
│   │   └── position.py                     (PositionSizer 基类)
│   │
│   ├── risk/                                # 风控模块
│   │   └── __init__.py                     (RiskManager, StopLoss, TakeProfit, etc.)
│   │
│   ├── execution/                           # 执行引擎
│   │   ├── backtest.py                     (BacktestEngine - 新架构)
│   │   └── executor.py                     (LiveExecutor)
│   │
│   ├── monitor/                             # 监控模块
│   │   ├── performance.py                  (PerformanceMonitor)
│   │   ├── realtime.py                     (RealtimeMonitor)
│   │   └── report.py                       (ReportGenerator)
│   │
│   ├── research/                            # 研究模块
│   │   ├── evaluation.py                   (FactorEvaluator)
│   │   └── attribution.py                  (AttributionAnalyzer)
│   │
│   ├── strategy/                            # 策略基类
│   │   └── base.py                         (SignalStrategy, CompositeStrategy)
│   │
│   ├── factors/                             # 因子计算 (预留目录)
│   │   ├── base/
│   │   └── impl/
│   │
│   ├── __init__.py
│   ├── backtest.py                         (保留 - 旧架构回测引擎)
│   ├── live_engine.py                      (保留 - 旧架构实盘引擎)
│   ├── strategy.py                         (保留 - 旧架构 QuantStrategy)
│   ├── signal_composer.py                  (保留 - 旧架构 SignalComposer)
│   ├── config.py                           (保留)
│   ├── database.py                         (保留)
│   ├── data_fetcher.py                     (保留)
│   ├── factor.py                           (保留)
│   ├── factor_hub.py                       (保留)
│   ├── factor_evaluator.py                 (保留)
│   ├── fee.py                              (保留)
│   ├── universe.py                         (保留)
│   ├── strategy_hub.py                     (保留)
│   ├── data_checker.py                     (保留)
│   └── data_validator.py                   (保留)
│
├── strategies/                              # 策略实现 (保持原样)
│   ├── momentum_strategy.py
│   ├── alpha101_strategy.py
│   ├── technical_strategy.py
│   ├── example_signal_strategy.py          (新架构示例)
│   └── config_signal_strategy.py           (新架构配置示例)
│
├── broker/                                  # 券商接口 (保持原样)
├── factors/                                 # 因子计算库 (保持原样)
├── scripts/                                 # 工具脚本 (保持原样)
├── tests/                                   # 测试 (保持原样)
├── config/                                  # 配置 (保持原样)
└── docs/                                    # 文档
    ├── ARCHITECTURE_V2.md
    ├── SIGNAL_STRATEGY_GUIDE.md
    └── ARCHITECTURE.md
```

---

## 模块职责说明

### screening/ (选股器)
- **职责**：从全市场股票中选出候选股票池
- **基类**：`IStockSelector`
- **实现**：`FactorRankSelector`, `MultiFactorSelector`, `FundamentalSelector`, `TrendBreakoutSelector`, `OversoldReboundSelector`, `ChipConcentrationSelector`

### timings/ (择时器)
- **职责**：判断市场环境，输出仓位系数 (0~1)，决定当前风险暴露
- **基类**：`ITimingGenerator`
- **实现**：`TrendTiming`, `VolatilityTiming`, `TrendVolatilityTiming`, `ComboTiming`, `MarketRegimeTiming`

### positioners/ (仓位分配器)
- **职责**：根据选股和择时信号，决定每个标的的持仓权重
- **基类**：`IPortfolioBuilder`
- **实现**：`EqualWeightBuilder`, `RiskParityBuilder`

### signals/ (信号组合器 - 新架构)
- **职责**：新架构 SignalStrategy 使用，组合选股、择时、风控信号
- **基类**：`IComposer`
- **实现**：`LayeredComposer`, `DirectComposer`, `WeightedComposer`, `VoteComposer`

### risk/ (风控模块)
- **职责**：对目标权重施加约束，限制风险
- **基类**：`IConstraint`
- **实现**：`MaxPositionConstraint`, `SingleWeightConstraint`, `StopLoss`, `TakeProfit`, `RiskManager`

### execution/ (执行引擎)
- **职责**：执行交易（回测或实盘）
- **实现**：`BacktestEngine`, `LiveExecutor`

### monitor/ (监控模块)
- **职责**：绩效分析、实时监控、报告生成
- **实现**：`PerformanceMonitor`, `RealtimeMonitor`, `ReportGenerator`

### research/ (研究模块)
- **职责**：因子评估、归因分析
- **实现**：`FactorEvaluator`, `AttributionAnalyzer`

---

## 两种架构并行支持

### 旧架构 (QuantStrategy)
```
from core.strategy import QuantStrategy
from core.screening import FactorRankSelector
from core.timings import TrendTiming
from core.positioners import EqualWeightBuilder

strategy = QuantStrategy(
    name="Momentum",
    selector=FactorRankSelector(factor_name="momentum_20"),
    timing=TrendTiming(),
    portfolio=EqualWeightBuilder(),
    top_n=30,
)
```

### 新架构 (SignalStrategy)
```
from core.strategy.base import SignalStrategy
from core.screening import FactorRankSelector
from core.signals.position import TrendPositionSizer
from core.signals import LayeredComposer
from core.risk import RiskManager

strategy = SignalStrategy(
    name="Momentum_v2",
    selector=FactorRankSelector(factor_name="momentum_20"),
    position_sizer=TrendPositionSizer(),
    composer=LayeredComposer(top_n=30),
    risk_manager=RiskManager(),
)
```

---

## 导入路径速查表

| 模块 | 导入方式 |
|------|---------|
| 选股器 | `from core.screening import FactorRankSelector` |
| 择时器 | `from core.timings import TrendTiming` |
| 仓位分配器 | `from core.positioners import EqualWeightBuilder` |
| 信号组合器 | `from core.signals import LayeredComposer` |
| 风控 | `from core.risk import RiskManager` |
| 回测 (新) | `from core.execution import BacktestEngine` |
| 回测 (旧) | `from core.backtest import BacktestEngine` |
| 监控 | `from core.monitor import PerformanceMonitor` |
| 研究 | `from core.research import FactorEvaluator` |

---

## 下一步建议

1. **测试兼容性**：运行现有策略，确认导入路径是否正常工作
2. **逐步迁移**：将新策略用 SignalStrategy 实现，旧策略保持 QuantStrategy
3. **文档完善**：根据实际使用情况更新文档
4. **因子目录**：后续可将 `factors/` 目录移动到 `core/factors/` 下

---

## 架构完成度

- ✅ screening/ - 迁移完成
- ✅ timings/ - 迁移完成
- ✅ positioners/ - 迁移完成
- ✅ signals/ - 已实现
- ✅ risk/ - 已实现
- ✅ execution/ - 已实现
- ✅ monitor/ - 已实现
- ✅ research/ - 已实现
- ✅ strategy/ - 已实现
- ⏸️ core/ 下单文件 - 保留兼容性，暂不移动
