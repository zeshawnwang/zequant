# ZEquant 文档索引

> 项目级文档导航，覆盖架构、方法论、迁移指南、代码审查等。
> 最后更新：2026-06-10（优化记录 + CODE_REVIEW）

---

## 文档清单

| 文档 | 说明 | 更新日期 |
|:----|:-----|:-------:|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构总览（模块分层、数据流、配置文件说明） | — |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | 重构后的积木式架构结构（core/ 目录组织） | — |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | 旧架构到新架构的完整迁移步骤 | — |
| [SIGNAL_STRATEGY_GUIDE.md](SIGNAL_STRATEGY_GUIDE.md) | 信号流驱动策略构建指南（选股/择时/仓位组合） | — |
| [METHODOLOGY.md](METHODOLOGY.md) | mss_dynamic 策略研究方法论沉淀（V1→V6 迭代经验） | 2026-05-27 |
| [CODE_REVIEW.md](CODE_REVIEW.md) | 全项目 Code Review 报告（架构/质量/因子/实盘审查） | 2026-06-10 |
| [2025-06-10-optimization-record.md](2025-06-10-optimization-record.md) | 2025-06-10 全面优化记录（12 项改动的实施与验证） | 2026-06-10 |

## 策略相关

| 文档 | 位置 |
|:----|:-----|
| [策略目录索引](../core/strategies/impl/INDEX.md) | `core/strategies/impl/INDEX.md` — 27 个策略的回测表现与 V7.1 实盘配置 |
| [策略工厂 README](../core/strategies/impl/_template/README.md) | `core/strategies/impl/_template/README.md` — 如何创建新策略 |

## 日常实验记录

| 位置 | 说明 |
|:-----|:-----|
| `daily/2026-05-27/` | V1→V6 主策略迭代实验（~400 次回测） |
| `daily/2026-06-02/` | V7 替换 c01_layered_d5 全面回测验证 |
| `daily/2026-06-09/` | V7.1 cooldown=5 状态冷却期实验 |
| `daily/2026-06-12/` | 因子 IR 时序分析权重优化实验 |

## 快速入口

- 最新实盘信号：`python3 -m live.signals.mss_dynamic --capital 50000`
- 全自动流水线：`python3 -m live.runner`
- 回测跟踪：`python3 daily/track_backtest.py`

---

*ZEquant 文档导航 — 需要补充新文档时请更新此索引*
