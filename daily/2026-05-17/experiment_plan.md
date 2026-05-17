# 2026-05-17 策略重评估计划

## 背景

2026-05-16 的调优实验（336个实验）发现了 Pipeline 回测与真实交易之间存在系统性偏差：

| 偏差项 | 旧Pipeline | 新Pipeline | 影响 |
|--------|-----------|-----------|------|
| ST过滤 | 无 | ✅ 按名称过滤 | 消除噪声选股 |
| 新股过滤 | 无 | ✅ 上市≥60天 | 消除炒作新股 |
| 涨跌停过滤 | 无（Look-ahead） | ✅ 按板块分级判定 | 避免无法买入 |
| 停牌过滤 | 无 | ✅ volume=0排除 | 避免无法交易 |
| 交易成本 | 0.12% flat | 0.20% flat（含印花税/佣金/滑点） | 更接近实盘 |

**结论：旧实验高估了策略收益和Sharpe。需要全部重跑。**

## 实验设计

### 实验1: MF核心参数扫描
- 变量: `top_n` ∈ {20, 30, 40, 50}, `rebal_freq` ∈ {3, 5, 10}, `min_hold_days` ∈ {3, 5, 10}
- 仓位类型: `rp`, `covrp`
- 固定: Default factors, `tx_cost=0.002`, `use_universe_filter=True`

### 实验2: Chip策略参数扫描
- 变量: `top_n` ∈ {20, 30, 50}, `rebal_freq` ∈ {5, 10, 20}
- 固定: Chip factors, `tx_cost=0.002`

### 实验3: 组合策略
- 将实验1最佳MF与实验2最佳Chip按不同比例组合
- 比例: 60/40, 50/50, 40/60

### 实验4: 动态策略
- 根据市场状态动态切换MF/Chip权重
- 使用 MarketStateSelector 规则

## 运行方式

```bash
# 实验1：MF参数扫描
python3 daily/2026-05-17/tuning_pipeline.py --experiment mf_params

# 实验2：Chip参数扫描
python3 daily/2026-05-17/tuning_pipeline.py --experiment chip_params

# 验证单策略
python3 daily/2026-05-17/tuning_pipeline.py --experiment validate --strategy mf_d10_rp
```

## 三区间验证

所有通过参数扫描的策略必须通过三区间验证：
1. 全区间 (2019-01 ~ 2026-04): Sharpe > 0.5
2. 修复牛OOS (2024-07 ~ 2026-04): Sharpe > 1.0
3. 2022熊市 (2022-01 ~ 2022-12): 年化 > 0%（不亏钱）

## 输出

结果将写入 `daily/2026-05-17/results/` 目录。
