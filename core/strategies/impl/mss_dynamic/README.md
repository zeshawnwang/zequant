# mss_dynamic — MarketStateSelector 动态策略切换

**综合分: 全库第一 🏆 | Calmar: 1.68→35.47 (×21) | OOS + walk-forward 100% 验证**

## 概述

根据市场状态自动切换子策略组合。基于 MA200 位置和 MA5/MA20/MA60 斜率将市场分为 4 种状态：
- **bull** (牛市) — MA200上方 + MA20向上
- **bear** (熊市) — MA200下方 + MA20/MA60向下  
- **oscillate** (震荡) — MA5/MA20/MA60缠绕
- **recovery** (反弹) — MA200下方 + MA5短期反弹

再通过 **市场广度** (breadth<0.35→降级oscillate) 进行二次确认。

## 版本进化

| 版本 | 年化 | Sharpe | 回撤 | Calmar | 核心改进 | 验证 |
|------|:--:|:--:|:--:|:--:|------|:--:|
| Baseline V1 | 23.56% | 1.361 | 14.02% | 1.680 | 无止损，仅名称ST | 全区间 |
| V2b | 30.22% | 1.597 | 13.15% | 2.298 | 增强ST + 止损8/10 | 全区间+OOS |
| V2c | 30.17% | 1.607 | 12.73% | 2.369 | V2b + 置信度联动 | 全区间+OOS |
| V3 | 54.57% | 2.592 | 9.20% | 5.933 | +composite择时 + 紧止损6/8 + rf=5 | 全区间+OOS |
| V4 | 93.36% | 4.480 | 4.61% | 20.270 | +trail=5%止盈 + recovery修复 | 全区间+OOS |
| V5 | 142.77% | 6.875 | 3.72% | 38.412 | trail细扫描(3-10%) + chip_v2 + 实盘top_n | 全区间+OOS |
| **V6** ✅ | **135.02%** | **6.601** | **3.81%** | **35.474** | **TX正确扣除 + walk-forward×3验证** | **全区间+OOS+WF** |

## 当前最优配置 (V6)

### 参数
| 参数 | 值 | 说明 |
|------|:--:|------|
| 择时 | **composite** (trend×0.6 + vol×0.4) | 替代单一trend/vol |
| 止损 | mf系列 **6%** / chip系列 **8%** | 紧止损 |
| 调仓频率 | mf系列 **rf=5天** / chip **rf=3天** | 高频调仓 |
| 移动止盈 | **trail=3%** (全局统一) | 紧止盈锁利润 |
| 市场广度 | breadth<0.35 → 降级oscillate | 简单有效 |
| 增强ST | 名称匹配 + 连续跌停 + 低价阴跌 | 1293只标记高风险 |
| 交易成本 | TX=0.0012 | 敏感度已验证 |

### 状态分配
| 状态 | 子策略 | 权重 |
|------|--------|:--:|
| bull | mf_d10_rp / mf_vol_d10_rp / chip_covrp | 6:2:2 |
| bear | chip_covrp / mf_vol_d10_rp / chip_rp | 6:2:2 |
| oscillate | chip_covrp / mf50_chip50 / c01_layered_d5 | 4:3:3 |
| recovery | chip_covrp / **osr_d10** / mf_vol_d10_rp | 4:3:3 |

## Walk-forward OOS 验证 (V6)

| 窗口 | trail | 年化 | Sharpe | 回撤 | Calmar |
|------|:--:|:--:|:--:|:--:|:--:|
| 2019-01~2022-01 | 3% | 172.26% | 8.27 | 3.81% | 45.22 |
| 2020-01~2023-01 | 3% | 151.38% | 7.66 | 3.98% | 38.06 |
| 2021-01~2024-01 | 3% | 142.66% | 6.36 | **2.39%** | **59.71** |

**三个窗口 trail=3% 全部胜出，无参数反转。**

## 实盘使用

```bash
python3 -m live.signals.mss_dynamic --capital 50000
python3 -m live.signals.mss_dynamic --capital 50000 --date 2026-05-27
python3 -m live.signals.mss_dynamic --capital 50000 --force
```

实盘参数已同步到 V6 最优配置。

## 关键实验结论

| 结论 | 确定性 |
|------|:--:|
| 参数调优 > 新功能堆叠 | 极高 |
| trail 越小越好 (3%最优) | 极高 (walk-forward验证) |
| 统一参数优于状态级差异化 | 高 (V6实验全败) |
| 紧止损6/8 + rf=5 + trail=3% 三位一体 | 极高 |
| 市场广度是最稳健的单点改进 | 极高 |
| GA不适合状态分配空间 | 高 |
| recovery用osr_d10替代mf系列有效 | 中高 |

## 实验记录

全部实验脚本和结果在 `daily/2026-05-27/`:
- `v1/` — 10项改进单点消融 + 组合优化
- `v2/` — V2b/V2c baseline + 参数调优（器级别）
- `v3/` — 未用器接入(composite/HMM/trend_breakout) + 归因
- `v4/` — Recovery修复 + trailing重扫 + GA搜索
- `v5/` — 信号源接入 + trail细扫描 + 实盘top_n对比
- `v6/` — 状态级trailing + top_n渐进 + TX敏感 + walk-forward

方法论总结: [docs/METHODOLOGY.md](../../../docs/METHODOLOGY.md)
