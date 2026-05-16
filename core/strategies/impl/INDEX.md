# ZEquant 策略目录索引

> 更新时间：2026-05-16
> 共 17 个已注册策略

## 总体排名（按Sharpe排序）

| 排名 | 策略 | 年化% | Sharpe | 回撤% | Calmar | 核心特色 | 策略路径 |
|:---:|:----|:----:|:-----:|:----:|:------:|:--------|:--------|
| 1 | ga_d10 | 33.59 | 1.435 | -16.33 | 2.057 | 全库最高Sharpe+最低回撤 | [ga_d10/](./ga_d10/) |
| 2 | chip_equal_d3 | 17.43 | 1.401 | -11.19 | 1.558 | 等权Chip最高Sharpe | [chip_equal_d3/](./chip_equal_d3/) |
| 3 | mf50_chipcovrp50_combo | 22.78 | 1.343 | -17.01 | 1.339 | 组合系列最高Sharpe | [mf50_chipcovrp50_combo/](./mf50_chipcovrp50_combo/) |
| 4 | mf_vol_d10_rp | 26.13 | 1.334 | -22.59 | 1.156 | MF+VolTiming择时 | [mf_vol_d10_rp/](./mf_vol_d10_rp/) |
| 5 | mf_d10_rp | 38.03 | 1.306 | -30.44 | 1.250 | 旗舰高收益低频 | [mf_d10_rp/](./mf_d10_rp/) |
| 6 | chip_covrp | 14.96 | 1.288 | -9.68 | 1.545 | 协方差风险平价最低回撤 | [chip_covrp/](./chip_covrp/) |
| 7 | osr_vol_eq_d10 | 21.52 | 1.280 | -16.43 | 1.310 | OSR+VolTiming风控 | [osr_vol_eq_d10/](./osr_vol_eq_d10/) |
| 8 | mf60_chip40_combo | 24.84 | 1.269 | -18.20 | 1.365 | 最佳年化/回撤组合 | [mf60_chip40_combo/](./mf60_chip40_combo/) |
| 9 | ga_d5 | 47.18 | 1.265 | -29.11 | 1.620 | 全库最高年化 | [ga_d5/](./ga_d5/) |
| 10 | mf50_chip50_combo | 22.29 | 1.257 | -16.60 | 1.343 | 组合最低回撤 | [mf50_chip50_combo/](./mf50_chip50_combo/) |
| 11 | osr_d10 | 23.55 | 1.164 | -26.39 | 0.892 | 超跌反弹旗舰 | [osr_d10/](./osr_d10/) |
| 12 | v1_ga_rp | 45.46 | 1.119 | -39.77 | 1.143 | V1最高收益 | [v1_ga_rp/](./v1_ga_rp/) |
| 13 | chip_vol_rp | 10.37 | 1.101 | -10.51 | 0.986 | 全库最低回撤 | [chip_vol_rp/](./chip_vol_rp/) |
| 14 | mf_trend_d5_rp | 17.16 | 1.083 | -14.42 | 1.190 | 趋势择时风控 | [mf_trend_d5_rp/](./mf_trend_d5_rp/) |
| 15 | chip_rp | 14.50 | 1.071 | -14.58 | 0.994 | 筹码集中基础版 | [chip_rp/](./chip_rp/) |
| 16 | v4_mf_rp | 26.93 | 0.976 | -28.71 | 0.938 | V4多因子 | [v4_mf_rp/](./v4_mf_rp/) |
| 17 | v4_mf_tv_rp | 21.87 | 0.918 | -16.32 | 1.340 | V4+趋势择时 | [v4_mf_tv_rp/](./v4_mf_tv_rp/) |

## 按风险等级分类

### 🟢 低风险（回撤<15%）
| 策略 | 年化% | 回撤% | Sharpe |
|:----|:----:|:----:|:-----:|
| chip_covrp | 14.96 | -9.68 | 1.288 |
| chip_vol_rp | 10.37 | -10.51 | 1.101 |
| chip_equal_d3 | 17.43 | -11.19 | 1.401 |
| mf_trend_d5_rp | 17.16 | -14.42 | 1.083 |
| chip_rp | 14.50 | -14.58 | 1.071 |

### 🟡 中风险（回撤15~25%）
| 策略 | 年化% | 回撤% | Sharpe |
|:----|:----:|:----:|:-----:|
| v4_mf_tv_rp | 21.87 | -16.32 | 0.918 |
| ga_d10 | 33.59 | -16.33 | 1.435 |
| osr_vol_eq_d10 | 21.52 | -16.43 | 1.280 |
| mf50_chip50_combo | 22.29 | -16.60 | 1.257 |
| mf50_chipcovrp50_combo | 22.78 | -17.01 | 1.343 |
| mf60_chip40_combo | 24.84 | -18.20 | 1.269 |
| mf_vol_d10_rp | 26.13 | -22.59 | 1.334 |

### 🔴 高风险（回撤>25%）
| 策略 | 年化% | 回撤% | Sharpe |
|:----|:----:|:----:|:-----:|
| osr_d10 | 23.55 | -26.39 | 1.164 |
| v4_mf_rp | 26.93 | -28.71 | 0.976 |
| ga_d5 | 47.18 | -29.11 | 1.265 |
| mf_d10_rp | 38.03 | -30.44 | 1.306 |
| v1_ga_rp | 45.46 | -39.77 | 1.119 |

## 按信号策略分类

### 多因子策略(MultiFactor)
| 策略 | 年化% | Sharpe | 回撤% |
|:----|:----:|:-----:|:----:|
| ga_d10 | 33.59 | 1.435 | -16.33 |
| ga_d5 | 47.18 | 1.265 | -29.11 |
| mf_vol_d10_rp | 26.13 | 1.334 | -22.59 |
| mf_d10_rp | 38.03 | 1.306 | -30.44 |
| mf_trend_d5_rp | 17.16 | 1.083 | -14.42 |
| v1_ga_rp | 45.46 | 1.119 | -39.77 |
| v4_mf_rp | 26.93 | 0.976 | -28.71 |
| v4_mf_tv_rp | 21.87 | 0.918 | -16.32 |

### 筹码集中策略(ChipConcentration)
| 策略 | 年化% | Sharpe | 回撤% |
|:----|:----:|:-----:|:----:|
| chip_equal_d3 | 17.43 | 1.401 | -11.19 |
| chip_covrp | 14.96 | 1.288 | -9.68 |
| chip_vol_rp | 10.37 | 1.101 | -10.51 |
| chip_rp | 14.50 | 1.071 | -14.58 |

### 超跌反弹策略(OversoldRebound)
| 策略 | 年化% | Sharpe | 回撤% |
|:----|:----:|:-----:|:----:|
| osr_vol_eq_d10 | 21.52 | 1.280 | -16.43 |
| osr_d10 | 23.55 | 1.164 | -26.39 |

### 组合策略(Combo)
| 策略 | 年化% | Sharpe | 回撤% |
|:----|:----:|:-----:|:----:|
| mf50_chipcovrp50_combo | 22.78 | 1.343 | -17.01 |
| mf60_chip40_combo | 24.84 | 1.269 | -18.20 |
| mf50_chip50_combo | 22.29 | 1.257 | -16.60 |

## 使用说明

从代码中调用策略的方法：
```python
from core.strategies.impl import build_mf_d10_rp
strategy = build_mf_d10_rp(top_n=40)

# 或直接使用策略名称
strategy = build_ga_d10(top_n=40)
strategy = build_ga_d5(top_n=40)
strategy = build_mf50_chipcovrp50_combo(top_n=40)
```

## 详细策略列表

| # | 策略名 | 年化 | Sharpe | 回撤 | 一句话 | README |
|:-:|:------|:---:|:-----:|:----:|:-------|:------|
| 1 | **ga_d10** | 33.59% | 1.435 | -16.33% | GA优化+10d+RP，全库最高Sharpe和最低回撤 | [📄](./ga_d10/README.md) |
| 2 | **ga_d5** | 47.18% | 1.265 | -29.11% | GA优化+5d+RP，全库最高年化 | [📄](./ga_d5/README.md) |
| 3 | **chip_covrp** | 14.96% | 1.288 | -9.68% | Chip+协方差风险平价，Chip系列最低回撤 | [📄](./chip_covrp/README.md) |
| 4 | **chip_equal_d3** | 17.43% | 1.401 | -11.19% | Chip+等权分配，Chip系列最高Sharpe | [📄](./chip_equal_d3/README.md) |
| 5 | **chip_rp** | 14.50% | 1.071 | -14.58% | Chip+无择时+RP，筹码集中基础版 | [📄](./chip_rp/README.md) |
| 6 | **chip_vol_rp** | 10.37% | 1.101 | -10.51% | Chip+VolTiming+RP，全库最低回撤 | [📄](./chip_vol_rp/README.md) |
| 7 | **mf50_chip50_combo** | 22.29% | 1.257 | -16.60% | MF_D10 50%+Chip_D3 50%，组合最低回撤 | [📄](./mf50_chip50_combo/README.md) |
| 8 | **mf50_chipcovrp50_combo** | 22.78% | 1.343 | -17.01% | MF_D10 50%+Chip_CovRP 50%，组合最高Sharpe | [📄](./mf50_chipcovrp50_combo/README.md) |
| 9 | **mf60_chip40_combo** | 24.84% | 1.269 | -18.20% | MF_D10 60%+Chip_D3 40%，最佳年化/回撤组合 | [📄](./mf60_chip40_combo/README.md) |
| 10 | **mf_d10_rp** | 38.03% | 1.306 | -30.44% | 多因子+10d+RP，旗舰高收益低频 | [📄](./mf_d10_rp/README.md) |
| 11 | **mf_trend_d5_rp** | 17.16% | 1.083 | -14.42% | 多因子+趋势择时+5d+RP | [📄](./mf_trend_d5_rp/README.md) |
| 12 | **mf_vol_d10_rp** | 26.13% | 1.334 | -22.59% | 多因子+VolTiming+10d+RP，最佳Sharpe择时版 | [📄](./mf_vol_d10_rp/README.md) |
| 13 | **osr_d10** | 23.55% | 1.164 | -26.39% | OSR+10d+RP，超跌反弹旗舰 | [📄](./osr_d10/README.md) |
| 14 | **osr_vol_eq_d10** | 21.52% | 1.280 | -16.43% | OSR+VolTiming+等权+10d，超跌+风控 | [📄](./osr_vol_eq_d10/README.md) |
| 15 | **v1_ga_rp** | 45.46% | 1.119 | -39.77% | V1 GA+技术因子裁量权+10d+RP，V1最高收益 | [📄](./v1_ga_rp/README.md) |
| 16 | **v4_mf_rp** | 26.93% | 0.976 | -28.71% | V4原始多因子+10d+RP | [📄](./v4_mf_rp/README.md) |
| 17 | **v4_mf_tv_rp** | 21.87% | 0.918 | -16.32% | V4多因子+趋势择时+10d+RP | [📄](./v4_mf_tv_rp/README.md) |

---

*ZEquant 策略库 — 持续更新中*
