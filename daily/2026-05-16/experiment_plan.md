# Zequant 策略实验规划总纲

> 创建日期：2026-05-16
> 目标：通过选股器×择时器×仓位分配器×调仓频率×信号组合器的交叉搭配，找到**最大回撤<20%、牛熊市兼容、年化最高**的策略。
> 最终产出：每个成功策略独立成文件夹落地到 `core/strategies/impl/{策略名}/`，含实现py + 讲解md + 参数配置文件。
> 评估标准：2019-01-02 ~ 2026-04-30 全区间回测，最大回撤 <20%，各市场窗口均为正收益。

---

## 目录

1. [实验方法论](#1-实验方法论)
2. [可用积木清单（完整）](#2-可用积木清单完整)
3. [实验矩阵——所有可能的交叉组合](#3-实验矩阵所有可能的交叉组合)
4. [实验类型 A——单变量扫描](#4-实验类型-a单变量扫描)
5. [实验类型 B——双变量交叉](#5-实验类型-b双变量交叉)
6. [实验类型 C——三变量组合](#6-实验类型-c三变量组合)
7. [实验类型 D——完整策略链路](#7-实验类型-d完整策略链路)
8. [实验类型 E——非最优器组合](#8-实验类型-e非最优器组合)
9. [核心约束——回撤<20%、牛熊兼容](#9-核心约束回撤20牛熊兼容)
10. [策略落地模板](#10-策略落地模板)
11. [实验记录表](#11-实验记录表)
12. [最终产出结构](#12-最终产出结构)

---

## 1. 实验方法论

### 1.1 核心思想

策略 = **选股器(Selector)** × **择时器(Timing)** × **仓位分配器(Positioner)** × **信号组合器(Composer)** × **调仓频率(Rebal)**

每个维度有多个选项，在V1~V4中只测试了少量组合。真实的最优策略很可能不是"每个器的最优选择"，而是某些**非最优的器搭配在一起反而更好**。

### 1.2 评估标准

每完成一个实验，记录以下指标：

| 指标 | 达标线 | 目标线 |
|:----|:------|:------|
| 年化收益 | >10% | >20% |
| Sharpe | >0.5 | >0.8 |
| **最大回撤** | **<30%** | **<20%** |
| Calmar | >0.3 | >0.8 |
| 各窗口正收益 | 5/7通过 | 7/7通过 |
| 市场窗口 | 2019修复牛 / 2020疫情 / 2021结构牛 / 2022熊市 / 2023震荡 / 2024反弹 / 2025至今 |

### 1.3 实验执行流程

1. 从"单变量扫描"开始（最快，找各维度的最佳选项）
2. 到"双变量交叉"（找最佳配对）
3. 到"三变量+完整链路"（最终策略组装）
4. 合格的策略落地到 `strategies/impl/{名称}/`

---

## 2. 可用积木清单（完整）

### 2.1 选股器 (Selector)

| ID | 类 | 来源 | 核心逻辑 | 已测？ |
|:--:|:---|:-----|:---------|:------:|
| S01 | `MultiFactorSelector(w, winsorize=0.01)` | `screening/impl/multi_factor.py` | 截面Z-Score加权+Winsorize去极值 | ✅ V3/V4 |
| S02 | `FactorRankSelector(factor_name, ascending)` | `screening/impl/factor_rank.py` | 单因子截面排名 | ❌ |
| S03 | `FundamentalSelector` | `screening/impl/fundamental.py` | 市盈率/市净率/ROE过滤排序 | ❌ |
| S04 | `TrendBreakoutSelector` | `screening/impl/momentum_breakout.py` | 均线多头+放量突破+MACD零轴上 | ❌ |
| S05 | `OversoldReboundSelector` | `screening/impl/momentum_breakout.py` | RSI超卖+价格低位+放量反弹 | ❌ |
| S06 | `ChipConcentrationSelector` | `screening/impl/momentum_breakout.py` | 成交量/波动率筛选筹码集中 | ❌ |

### 2.2 择时器 (Timing) — 决定**仓位系数**

| ID | 类 | 输出 | 核心逻辑 | 已测？ |
|:--:|:---|:----|:---------|:------:|
| T01 | `TrendTiming` | 每只0~1连续分 | MACD+动量5/20+RSI均值 | ❌ |
| T02 | `VolatilityTiming` | 仅SELL信号 | vol>0.30卖出，不产生BUY | ❌ |
| T03 | `TrendVolatilityTiming` | 趋势分+高波动卖出 | T01+T02复合 | ✅ V3/V4 |
| T04 | `MarketRegimeTiming` | 大盘牛/熊/震荡 | 均线多头+60日线占比 | ✅ V4 |
| T05 | `CompositeTiming(timings, mode)` | 投票/加权结果 | 多择时器融合 | ❌ |
| T06 | `None`（无择时） | 仓位系数=1.0 | 永远满仓 | ✅ V4 |

### 2.3 仓位分配器 (Positioner) — 决定**个股权重**

| ID | 类 | 类型 | 核心逻辑 | 已测？ |
|:--:|:---|:----|:---------|:------:|
| P01 | `RPPortfolioWeights(top_n, min_hold)` | 底层numpy | 波动率倒数加权 | ✅ V1~V4 |
| P02 | `RiskParityBuilder` | IPortfolioBuilder | 协方差矩阵真实风险平价 | ❌ |
| P03 | `HysteresisAllocator(enable=False)` | 底层numpy | 等价于P01 | ✅ |
| P04 | `HysteresisAllocator(enable=True)` | 底层numpy | P01+大仓位惯性过滤 | ✅ V2 |
| P05 | `EqualWeightBuilder` | IPortfolioBuilder | 等权 | ❌ |
| P06 | `TrendPositionSizer` | ISizer | MACD+动量+RSI→0~1仓位 | ❌ |
| P07 | `VolatilityPositionSizer` | ISizer | 波动率倒数定仓位 | ❌ |
| P08 | `CompositePositionSizer` | ISizer | 多sizer融合 | ❌ |
| P09 | `RiskParityPositionSizer` | ISizer | 自定义风险平价 | ❌ |
| P10 | `FixedPositionSizer` | ISizer | 固定仓位 | ❌ |

### 2.4 信号组合器 (Composer) — 决定**选股×择时的合成方式**

| ID | 类 | 逻辑 | 已测？ |
|:--:|:---|:----|:------:|
| C01 | `LayeredComposer` | 先选股→择时定仓位→分配 | ❌ |
| C02 | `DirectComposer` | 选股和择时直接相乘 | ❌ |
| C03 | `WeightedComposer` | 多信号加权求和 | ❌ |
| C04 | `VoteComposer` | 多信号投票 | ❌ |
| C05 | `None`（手写dot） | `np.dot(factor_z, w)`直接排序 | ✅ V1~V4 |

### 2.5 调仓频率 (Rebal)

| ID | 频率 | 调仓周期 | 年化调仓次数 | 已测？ |
|:--:|:----|:--------|:----------:|:------:|
| R01 | 每日(1d) | 1交易日 | ~240 | ✅ |
| R02 | 隔日(2d) | 2交易日 | ~120 | ❌ |
| R03 | 周频(3d) | 3交易日 | ~80 | ✅ |
| R04 | 周频(5d) | 5交易日 | ~48 | ❌ |
| R05 | 双周(10d) | 10交易日 | ~24 | ❌ |
| R06 | 月频(21d) | 21交易日 | ~12 | ❌ |
| R07 | 条件调仓 | 换手>阈值才调 | 不定 | ❌ |

### 2.6 因子权重 (Weight)

| ID | 来源 | 已测？ |
|:--:|:-----|:------:|
| W01 | V1 GA优化权重（48非零因子） | ✅ |
| W02 | 等权（50因子各1/50） | ❌ |
| W03 | `MultiFactorSelector.from_registry(db)` 数据库IR权重 | ❌ |
| W04 | 全306因子L1稀疏化 | ❌ |
| W05 | 单因子（如 momentum_20） | ❌ |

### 2.7 风控层 (Risk)

| ID | 方式 | 已测？ |
|:--:|:-----|:------:|
| K01 | 无（仅靠择时降仓） | ✅ |
| K02 | `FeeCalculator` 精确逐笔费率 | ❌ |
| K03 | `RiskManager` 涨跌停+ST过滤 | ❌ |
| K04 | 持仓天数止损（30天强制卖出） | ❌ |
| K05 | 行业中性化约束 | ❌ |
| K06 | 最大单个权重限仓 | ❌ |

---

## 3. 实验矩阵——所有可能的交叉组合

理论上：**6选股器 × 6择时器 × 10分配器 × 5组合器 × 7频率 × 5权重** = **63,000** 种组合。

但很多组合没有意义（如择时器T02只出SELL信号，不能单独用）。项目实践可做的大概**200~300组**有意义的组合。

### 按复杂度排序

```
类型A: 单变量扫描 — 固定其他维度，只变1个          → 约30个实验
类型B: 双变量交叉 — 固定其他，变2个                 → 约80个实验
类型C: 三变量组合 — 三自由度的关键区域探索            → 约60个实验
类型D: 完整策略链路 — 最终组装的整策略               → 约20个实验
类型E: 非最优器组合 — 看似"次优"的器搭在一起          → 约20个实验
```

---

## 4. 实验类型 A——单变量扫描

固定条件：MF(W01) + 无择时(T06) + RPPortfolioWeights(P01) + 手写dot(C05) + 周频3d(R03)

### 实验A01~A06：选股器扫描

| 编号 | 变量 | 固定项 |
|:----:|:----|:-------|
| A01 | S01 `MultiFactorSelector` | T06 + P01 + C05 + R03 + W01 | 基准 |
| A02 | S02 `FactorRankSelector(momentum_20)` | 同上 | 动量单因子 |
| A03 | S03 `FundamentalSelector` | 同上 | 基本面 |
| A04 | S04 `TrendBreakoutSelector` | 同上 | 技术突破 |
| A05 | S05 `OversoldReboundSelector` | 同上 | 超跌反弹 |
| A06 | S06 `ChipConcentrationSelector` | 同上 | 筹码集中 |

### 实验A07~A12：择时器扫描

固定：MF(S01) + P01 + C05 + R03 + W01

| 编号 | 变量 |
|:----:|:----|
| A07 | T06 无择时（基准） |
| A08 | T01 `TrendTiming` |
| A09 | T02 `VolatilityTiming` |
| A10 | T03 `TrendVolatilityTiming` |
| A11 | T04 `MarketRegimeTiming` |
| A12 | T05 `CompositeTiming(T01+T02, vote)` |

### 实验A13~A19：仓位分配器扫描

固定：MF(S01) + T06 + C05 + R03 + W01

| 编号 | 变量 |
|:----:|:----|
| A13 | P05 `EqualWeightBuilder` |
| A14 | P01 `RPPortfolioWeights`（基准） |
| A15 | P02 `RiskParityBuilder`（协方差真实风险平价） |
| A16 | P03 `HysteresisAllocator(enable=False)` |
| A17 | P04 `HysteresisAllocator(enable=True)` |
| A18 | P06 `TrendPositionSizer`（注：与择时器不同，它输出连续仓位系数） |
| A19 | P07 `VolatilityPositionSizer` |

### 实验A20~A24：调仓频率扫描

固定：MF(S01) + T06 + P01 + C05 + W01

| 编号 | 变量 |
|:----:|:----|
| A20 | R02 隔日2d |
| A21 | R03 周频3d（基准） |
| A22 | R04 周频5d |
| A23 | R05 双周10d |
| A24 | R06 月频21d |

### 实验A25~A27：因子权重扫描

固定：S01 + T06 + P01 + C05 + R03

| 编号 | 变量 |
|:----:|:----|
| A25 | W02 等权（50因子各1/50） |
| A26 | W03 `from_registry` 数据库IR权重 |
| A27 | W05 单因子 momentum_20 |

---

## 5. 实验类型 B——双变量交叉

### 实验B01~B04：选股器 × 调仓频率

| 编号 | 选股器 | 频率 | 固定 |
|:----:|:------|:----|:-----|
| B01 | S01 MF | R02~R06 | T06+P01+C05 |
| B02 | S04 TrendBreakout | R02~R06 | 同上 |
| B03 | S05 OversoldRebound | R02~R06 | 同上 |
| B04 | S06 ChipConcentration | R02~R06 | 同上 |

### 实验B05~B08：择时器 × 仓位分配器

| 编号 | 择时器 | 分配器 | 固定 |
|:----:|:------|:------|:-----|
| B05 | T01 TrendTiming | P02/P05/P06/P07 | S01+C05+R03+W01 |
| B06 | T03 TVTiming | P02/P05/P06/P07 | 同上 |
| B07 | T04 MR| P02/P05/P06/P07 | 同上 |
| B08 | T05 Composite| P02/P05/P06/P07 | 同上 |

### 实验B09~B10：信号组合器 × 选股器

| 编号 | 组合器 | 选股器 | 固定 |
|:----:|:------|:------|:-----|
| B09 | C01/C02/C03/C04 | S01 MF | T06+P01+R03+W01 |
| B10 | C01/C02/C03/C04 | S04 TrendBreakout | 同上 |

---

## 6. 实验类型 C——三变量组合

这是最有价值的区域——3个自由度，覆盖绝大部分有意义的中等复杂度策略。

### 实验C01~C05：选股器 × 择时器 × 分配器（基础链路）

固定：C05手写dot + R03周频3d + W01

| 编号 | 选股器 | 择时器 | 分配器 |
|:----:|:------|:------|:------|
| C01 | S01 MF | T01 TrendTiming | P01 RPPortfolio |
| C02 | S01 MF | T04 MarketRegime | P02 RiskParityBuilder |
| C03 | S04 TrendBreakout | T01 TrendTiming | P06 TrendPositionSizer |
| C04 | S05 OversoldRebound | T03 TVTiming | P07 VolatilityPositionSizer |
| C05 | S06 ChipConcentration | T05 Composite(T01+T02) | P01 RPPortfolio |

### 实验C06~C10：选股器 × 频率 × 分配器

固定：T06无择时 + C05 + W01

| 编号 | 选股器 | 频率 | 分配器 |
|:----:|:------|:----|:------|
| C06 | S01 MF | R02(2d) | P02 RiskParity |
| C07 | S01 MF | R05(10d) | P05 EqualWeight |
| C08 | S04 TrendBreakout | R03(3d) | P01 RPPortfolio |
| C09 | S04 TrendBreakout | R05(10d) | P05 EqualWeight |
| C10 | S05 OversoldRebound | R06(21d) | P05 EqualWeight |

### 实验C11~C15：择时器 × 频率 × 分配器（风控组合）

固定：S01 MF + C05 + W01

| 编号 | 择时器 | 频率 | 分配器 | 预期效果 |
|:----:|:------|:----|:------|:---------|
| C11 | T01 TrendTiming | R02(2d) | P06 TrendSizer | 双趋势过滤，回撤最小 |
| C12 | T04 MarketRegime | R05(10d) | P02 RiskParity | 大盘择时+低频+真风险平价 |
| C13 | T03 TVTiming | R03(3d) | P07 VolSizer | 双波动风控 |
| C14 | T05 Composite(T01+T04) | R04(5d) | P01 RPPortfolio | 趋势+大盘双择时 |
| C15 | T02 VolatilityTiming | R03(3d) | P08 CompositeSizer | 纯风控型，回撤最低 |

---

## 7. 实验类型 D——完整策略链路

使用完整的 `SignalStrategy` 链路落地，结果可直接部署。

### 实验D01：MF+TrendTiming+RiskParityBuilder (回撤控制型)

```python
selector = MultiFactorSelector(v1_weights)
timing = TrendTiming(directly on stocks for position ratio)
# TrendTiming 输出0~1仓位系数
# → 选股40只 → 仓位系数 × 0.8 → RiskParity分配
```

预期：年化>15%，回撤<25%

### 实验D02：MF+MarketRegime+RiskParityBuilder (牛熊自适应)

```python
selector = MultiFactorSelector(v1_weights)
timing = MarketRegimeTiming()
# 牛市→1.0仓位  震荡→0.6  熊市→0.3
# → 选股40只 → 仓位系数分配
```

预期：年化>20%，熊市回撤<20%

### 实验D03：TrendBreakout+TrendTiming (全技术面策略)

```
选股器:    TrendBreakoutSelector (形态突破)
择时器:    TrendTiming (趋势判断)
组合器:    LayeredComposer
分配器:    RPPortfolioWeights
频率:      周频3d
```

### 实验D04：OversoldRebound+TVTiming (抄底风控型)

```
选股器:    OversoldReboundSelector
择时器:    TrendVolatilityTiming (防止抄在山顶)
组合器:    DirectComposer
分配器:    VolatilityPositionSizer (超跌后波动大→轻仓)
频率:      周频5d（更慢，等反弹确认）
```

### 实验D05：Composite(TrendBreakout+MF) + MarketRegime + RiskParity (双信号混合)

```
信号A:    TrendBreakoutSelector → 形态评分
信号B:    MultiFactorSelector → 多因子评分
组合器:    WeightedComposer(各0.5权重) → 综合排名
择时器:    MarketRegimeTiming → 大盘仓位系数
分配器:    RiskParityBuilder
频率:      周频3d
```

### 实验D06：ChipConcentration+MF+CompositeTiming+CompositeSizer (最复杂)

```
选股器A:  ChipConcentrationSelector
选股器B:  MultiFactorSelector
择时器A:  TrendTiming
择时器B:  VolatilityTiming (仅SELL)
择时组合:  CompositeTiming(Trend+Vol, mode="weighted")
选股组合:  VoteComposer(取交集)
仓位分配:  CompositePositionSizer(mode="min")
组合器:    LayeredComposer
频率:      周频3d
```

---

## 8. 实验类型 E——组合不传递性（局部最优 ≠ 全局最优）

> **核心思想**：假设有选股器1/2、择时器a/b，共1a/1b/2a/2b四种组合。单独看，选股器1强于2，择时器a强于b。但组合起来，**1a不一定是最优的**——可能1b>1a，甚至2a>1a。
>
> 原因：器与器之间有交互效应。一个强选股器可能和强择时器"打架"（都过度自信导致同向共振），而强选股器+弱择时器反而温和互补。
>
> **这就是本类型要探索的：打破"每个维度取最优 → 组合最优"的直觉。** 以下实验专门测试那些"单独看不是最优，但搭配起来可能更好"的跨维度组合。

### 为什么单变量扫描不够

Type A（单变量扫描）测试的是"固定其他维度不变，只变一个器"。这会告诉你每个器的最佳选择，但**无法告诉你这些最佳选择组合起来会发生什么**。

举一个真实例子（来自V1~V4）：
```
A01 已测：MF(最优选股器) + 无择时 + RP → 年化5.41%，回撤-84%
A09 已测：MF(最优选股器) + TVTiming(最优择时器? ) + RP → 年化1.12%，回撤-19%
A08 需测：MF(最优选股器) + TrendTiming(温和择时器) + RP → ❓可能年化回到10%+且回撤<20%
```

**两个单独最优的器(Today: MF最强，TV择时回撤最小)组合起来(1a)，不一定优于MF+温和择时(1b)。**

### 实验 E01~E06：选股器最佳 × 择时器次优 — 强选股 × 温和择时

变量：选股器固定为A01最强的(MF)，择时器从最优到次优变化。

| 编号 | 选股器(A01最优) | 择时器（降序） | 分配器 | 频率 | 疑问 |
|:----:|:--------------|:-------------|:------|:----|:-----|
| **E01** | MF(V1权重) | **无择时**(最强择时? 年化最高) | RP | 3d | 最强选股+最强收益，但回撤-84% |
| **E02** | MF(V1权重) | **TrendTiming**(温和择时) | RP | 3d | 最强选股+温和择时，回撤能否<20%? |
| **E03** | MF(V1权重) | **TrendVolTiming**(激进择时) | RP | 3d | 回撤-19%但年化仅1%，overshoot |
| **E04** | MF(V1权重) | **MarketRegime**(低频大盘择时) | RP | 3d | 大盘择时不看个股，和MF互补？ |
| **E05** | MF(V1权重) | **VolTiming**（只出SELL） | RP | 3d | 纯风控式择时，不干扰MF选股 |
| **E06** | MF(V1权重) | **TrendSizer**(连续仓位) | RP | 3d | 连续仓位vs三档仓位，哪个更优？ |

**预判**：E02(MF+TrendTiming)可能是甜点位——TrendTiming只有趋势信号没有波动率强制卖出，仓位不至于降到24%。

### 实验 E07~E12：择时器最佳 × 选股器次优 — 强择时 × 温和选股

变量：择时器固定为A09最强的(TVTiming)或A07最强的(TrendTiming)，选股器从最优到次优变化。

| 编号 | 选股器（降序） | 择时器(A09最优) | 分配器 | 频率 | 疑问 |
|:----:|:-------------|:--------------|:------|:----|:-----|
| **E07** | **MF(V1权重)**最强选股 | TrendVolTiming | RP | 3d | 强选股+强择时，打架还是互补？ |
| **E08** | **FactorRank(动量20)**激进单因子 | TrendVolTiming | RP | 3d | 激进选股+激进择时？ |
| **E09** | **TrendBreakout**技术形态 | TrendVolTiming | RP | 3d | 技术形态选股+趋势择时，逻辑一致？ |
| **E10** | **OversoldRebound**超跌 | TrendVolTiming | RP | 3d | 抄底+保守择时，矛盾？ |
| **E11** | **ChipConcentration**筹码集中 | TrendVolTiming | RP | 3d | 筹码集中+趋势择时 |
| **E12** | **Fundamental**基本面 | TrendVolTiming | RP | 3d | 基本面+趋势择时 |

**预判**：E09(TrendBreakout+TVTiming)可能效果奇特——技术突破选出的股票本身就偏趋势，再叠加趋势择时，可能形成信号共振；而E10可能会有对冲效果（超跌选出的股票在TV视角下可能被卖出）。

### 实验 E13~E16：分配器最佳 × (选股×择时)非最优组合

变量：分配器从最优(RP)变到次优，同时选股和择时都用非最优搭配。

| 编号 | 选股器 | 择时器 | 分配器 | 频率 | 疑问 |
|:----:|:------|:------|:------|:----|:-----|
| **E13** | MF(最优) | TrendTiming(温和) | **EqualWeight(次优)** | 3d | 强选股+温和择时+等权，比RP差多少？ |
| **E14** | FactorRank(次优) | MarketRegime(次优) | **RiskParityBuilder(真RP)** | 5d | 双次优+真风险平价，以分配器质量弥补？ |
| **E15** | TrendBreakout(次优) | TrendTiming(温和) | **EqualWeight** | 5d | 温和的技术面策略，低换手+等权 |
| **E16** | OversoldRebound(次优) | VolTiming(风控) | **EqualWeight** | 10d | 完全的"弱器集合"，低频+等权 |

**预判**：E15 可能是一匹黑马——技术突破选股频率本就不高（不是每天都有突破），等权分配下换手极低，加上温和择时，可能回撤很小。

### 实验 E17~E21：多维度同时非最优 — "第二好"大乱斗

完全不取最优，全维度都用"第二好"甚至"第三好"。

| 编号 | 选股器 | 择时器 | 分配器 | 频率 | 思路 |
|:----:|:------|:------|:------|:----|:-----|
| **E17** | FactorRank(2nd) | TrendTiming(2nd) | EqualWeight(2nd) | 5d(2nd) | 全维度第二好，可能是最稳的 |
| **E18** | TrendBreakout(5th?) | MarketRegime(3rd) | RiskParityBuilder(1st?) | 10d(6th) | 1好5差+1好1差 |
| **E19** | ChipConcentration(6th) | VolTiming(5th) | RP(1st) | 3d | 最差选股+最差择时+最好分配 |
| **E20** | OversoldRebound(5th) | TrendTiming(2nd) | TrendSizer(?) | 3d | 超跌选股+趋势择时+趋势仓位 |
| **E21** | MultiFactorSelector(1st) | 无择时(1st) | **Hysteresis(迟滞)** | 3d | 所有最优+迟滞分配器(比RP多一层过滤) |

**预判**：E17可能是最大惊喜——没有最突出的器，但所有器都是稳健的第二选择，组合起来可能最稳定。E21是最有的放矢的——V1本身就是最优，加一层迟滞分配器可能降低微调噪声。

---

**Type E 的核心原则总结：**

```
单维度最优  ≠  多维度组合最优
    因为：
    1. 交互效应：强选股×强择时可能同向共振放大风险
    2. 饱和效应：选股已经够强了，择时再强也是白费
    3. 补偿效应：一个维度的弱点可以被另一个维度的优点弥补
    4. 频次错配：高选股频率(日频) + 低择时频率(周频)可能不协调

Type E 专门探索这些"非直觉"的组合空间。
```

---

## 9. 核心约束——回撤<20%、牛熊兼容

这是最难的部分。V1年化45%但回撤49%，V4择时回撤19%但年化仅1%。中间地带需要精细探索。

### 9.1 七个市场窗口的定义

| 窗口 | 市场类型 | 沪深300涨跌 | 策略应表现 |
|:----|:--------|:-----------|:----------|
| 2019_修复牛 | 强牛市 | +36% | ≥+15% |
| 2020_疫情冲击+反弹 | 高波动 | +27% | ≥+10%（波动中存活） |
| 2021_结构牛 | 震荡偏强 | -5% | ≥+5%（结构行情） |
| 2022_熊市 | 强熊市 | -21% | ≥-10%（不亏或小亏） |
| 2023_震荡修复 | 弱修复 | -11% | ≥+5% |
| 2024_反弹 | 反弹 | +16% | ≥+10% |
| 2025_至今 | 震荡 | 平 | ≥+5% |

**达标条件**：7个窗口中至少5个正收益，且最大单窗口亏损<-15%的不能超过1个。

### 9.2 降低回撤的核心手段

| 手段 | 预期降回撤幅度 | 对收益的影响 |
|:----|:-------------:|:-----------:|
| 选股器改为保守型（技术突破而非多因子） | 5-10% | -10%~-20% |
| 择时器降仓（趋势不明确时只半仓） | 15-30% | -30%~-50% |
| 调仓频率降低（月频而非周频） | 5-10% | -10%~-20% |
| 仓位分配器从风险平价改为等权 | 2-5% | -5%~-10% |
| 信号组合器从Layered改为Direct | 0-5% | 0~-5% |
| 增加风控层（涨跌停/行业中性） | 2-5% | -2%~-5% |
| 多策略组合（不依赖单一信号） | 10-20% | +5%~+15%（互补效应） |

**最有效的降回撤手段是"择时器降仓"和"多策略组合"**。

---

## 10. 策略落地模板

每个成功策略在 `core/strategies/impl/{策略名}/` 下有三个文件。

### 10.1 文件结构

```
core/strategies/impl/
├── __init__.py
├── hub.py
├── v1_pipeline.py
├── momentum_strategy.py
├── ...
│
├── {strategy_name}/                           ← 策略独立文件夹
│   ├── __init__.py                            ← 导出 build_{name}()
│   ├── build.py                               ← 策略组装代码
│   ├── README.md                              ← 策略全程讲解
│   └── config.json                            ← 各个器的参数配置
```

### 10.2 `build.py` 模板

```python
"""策略名 — 一句话说明策略设计思路。

适用市场环境：xxx
回撤控制手段：xxx
预期年化/Sharpe/回撤：xx%/x.xx/xx%
"""
from __future__ import annotations

from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....timings import TrendTiming
from ....positioners import RPPortfolioWeights
from ....signals import LayeredComposer, MaxSingleWeightConstraint

import json, os


def build_{name}(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_file = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_file) as f:
        cfg = json.load(f)

    selector = MultiFactorSelector(
        weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01),
        top_n=top_n,
    )

    position_sizer = TrendTiming(
        buy_threshold=cfg["timing"].get("buy_threshold", 0.6),
        sell_threshold=cfg["timing"].get("sell_threshold", 0.4),
    )

    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.10),
        ],
    )

    # 注：仓位分配器在 SignalStrategy 的 composer 内部使用
    # 如果是底层分配器(RPPortfolioWeights)，需在回测时注入

    return SignalStrategy(
        name="{name}",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        top_n=top_n,
    )
```

### 10.3 `config.json` 模板

```json
{
  "strategy": {
    "name": "策略名",
    "version": "1.0",
    "description": "一句话说明",
    "target_market": "bull_bear_compatible",
    "created": "2026-05-16"
  },
  "selector": {
    "type": "MultiFactorSelector",
    "weight_source": "v1_ga_optimized",
    "winsorize": 0.01,
    "weights": {}
  },
  "timing": {
    "type": "TrendTiming",
    "buy_threshold": 0.6,
    "sell_threshold": 0.4,
    "note": "0~1仓位系数"
  },
  "positioner": {
    "type": "RPPortfolioWeights",
    "top_n": 40,
    "min_hold_days": 5,
    "vol_lookback": 20
  },
  "composer": {
    "type": "LayeredComposer",
    "top_n": 40,
    "constraints": {
      "max_single_weight": 0.10,
      "reserve_cash_ratio": 0.05
    }
  },
  "rebal": {
    "freq_days": 3,
    "type": "fixed_interval"
  },
  "risk": {
    "fee_calc": {
      "stamp_tax": 0.0005,
      "commission": 0.0003,
      "slippage": 0.0005,
      "min_commission": 5
    },
    "stop_loss": null,
    "max_position_pct": 0.15
  },
  "expected": {
    "annual_return": 0.0,
    "sharpe": 0.0,
    "max_drawdown": 0.0,
    "notes": "最终回测结果填这里"
  }
}
```

### 10.4 `README.md` 模板

```markdown
# {策略名}

## 设计思路
- 选股器为什么选这个：xxx
- 择时器为什么选这个：xxx
- 为什么是这个组合：xxx

## 各个器的参数和调优过程
- 选股器参数调优记录
- 择时器参数调优记录
- ...

## 实验过程中的坑和发现
- 什么组合试过但不行：xxx
- 什么参数会导致崩溃：xxx

## 回测结果
| 市场区间 | 年化 | 回撤 | Sharpe | 胜率 |
|---------|:---:|:---:|:-----:|:---:|
| 2019_修复牛 | | | | |
| ... | | | | |
| 全区间 | | | | |

## 优缺点和适用场景
- 优点：
- 缺点：
- 最适合的市场环境：

## 改进方向
```

---

## 11. 实验记录表

每个实验完成后，在 `daily/2026-05-16/results/` 下创建一个文件：

```
daily/2026-05-16/results/A01_MultiFactorSelector.json    ← A01的实验结果
daily/2026-05-16/results/B03_TrendBreakout_x_Rebal.json  ← B03的实验结果
daily/2026-05-16/results/C05_Complex.json                ← C05的实验结果
daily/2026-05-16/results/SUMMARY.md                      ← 所有实验结果汇总
```

### 汇总表格式

| 实验 | 选股器 | 择时器 | 分配器 | 频率 | 年化% | Sharpe | 回撤% | Calmar | 窗口通过 | 状态 |
|:---:|:-----:|:-----:|:-----:|:---:|:----:|:-----:|:----:|:-----:|:-------:|:---:|
| A01 | MF | 无 | RP | 3d | 5.41 | 0.072 | -84.2 | 0.064 | 3/7 | ✅ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

达标（年化>15%&回撤<20%）的行用 🏆 标记。

---

## 12. 最终产出结构

### 12.1 实验期产出

```
daily/2026-05-16/
├── experiment_plan.md                  ← 本文档
├── results/
│   ├── A01_MultiFactorSelector.json
│   ├── A02_FactorRankSelector.json
│   ├── ...
│   ├── SUMMARY.md                      ← 汇总表
│   └── pipeline_v5_v6_v7...py          ← 各组管道脚本
└── notes/
    ├── selector_deep_dive.md           ← 选股器专项分析
    ├── timing_analysis.md              ← 择时器专项分析
    └── positioner_comparison.md        ← 分配器专项分析
```

### 12.2 策略落地产出

```
core/strategies/impl/
├── __init__.py
├── hub.py
├── momentum_strategy.py
├── ...
│
├── def_mf_trend_rp/                    ← 防御型：MF+趋势择时+风险平价
│   ├── __init__.py
│   ├── build.py
│   ├── README.md
│   └── config.json
│
├── bal_mf_mr_riskparity/               ← 均衡型：MF+大盘择时+真风险平价
│   ├── __init__.py
│   ├── build.py
│   ├── README.md
│   └── config.json
│
├── agg_tb_trend_equal/                 ← 进攻型：技术突破+趋势择时+等权
│   ├── __init__.py
│   ├── build.py
│   ├── README.md
│   └── config.json
│
├── hedged_osr_tv_vol/                  ← 抄底型：超跌反弹+TV择时+波动仓位
│   ├── __init__.py
│   ├── build.py
│   ├── README.md
│   └── config.json
│
└── hybrid_mf_tb_mr_sizer/              ← 混合型：MF+突破+大盘择时+多sizer
    ├── __init__.py
    ├── build.py
    ├── README.md
    └── config.json
```

每个策略文件夹对应一个完整可运行的策略方案，包含复现所需的所有配置和说明文档。

---

## 附录：全版本回顾（截至2026-05-16）

| 版本 | 实验内容 | 最佳年化 | 最佳Sharpe | 最佳回撤 | 结论 |
|:---:|:---------|:-------:|:---------:|:--------:|:----|
| V1 | GA权重+风险平价+周频 | **45.46%** | **1.187** | -49.26% | 最佳收益，回撤不达标 |
| V2 | 换手惩罚+迟滞分配器 | 11.68% | 0.412 | -64.56% | GA有Bug，结论不可靠 |
| V3 | MF vs TV × 频率 × 分配器 | 52.89% | 0.330 | -81.25% | MF远优于TV；迟滞微弱 |
| V4 | 择时器对比 | 1.12% | 0.114 | **-19.33%** | TV降回撤最好但收益低 |

**核心结论**：单个器的最佳选择（如V1的MF+GA）会给最佳收益但回撤太大。**V5+的目标是找到一个不靠牺牲80%收益来降低回撤的平衡点**——初步判断方向是：强选股器(MF) + 温和择时器(TrendTiming) + 真风险平价(RiskParityBuilder) + 中等频率(3~5d)，加上可能的双信号融合同步降低回撤。

---

*文档生成: 2026-05-16*
*实验规划总纲，逐条执行。*
