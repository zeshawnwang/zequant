# ZEquant 优化记录 (2025-06-10)

按计划完成 12 步全面优化，涵盖配置校验、性能、代码质量、架构拆分。

---

## 1. Config Pydantic 校验层

**涉及文件：**
- `core/config_model.py` — 新增，定义 `ZeQuantConfig` + 子模型
- `core/config.py` — `load_config()` 返回 `ZeQuantConfig`（支持 `cfg["key"]` 向后兼容）
- `requirements.txt` — 新增 `pydantic>=2.0.0`

**效果：** 配置加载后过 Pydantic schema，类型错误/缺失值启动即报错，不再运行时崩溃。

---

## 2. DuckDB 参数自动化

**涉及文件：** `core/database.py:_configure_duckdb()`

- 线程数从硬编码 `8` 改为 `os.cpu_count()`
- 内存限制从硬编码 `4GB` 改为物理内存 50%（上限 8GB）
- `psutil` 不可用时降级 2GB 并记录 `logger.warning`

---

## 3. SQL 参数化

**涉及文件：**
- `live/data_updater.py` — 2 处 `LIKE '{p}%'` 拼接改为 DuckDB `~`/`!~` REGEXP 参数化
- `core/database.py` — `ensure_factor_columns` 列名校验从弱 `isalnum()` 改为严格正则 `^[a-zA-Z_][a-zA-Z0-9_]*$`

---

## 4. Pipeline 前瞻偏差修复

**涉及文件：** `core/strategies/pipeline.py:load()`

`load()` 获取股票代码时按 `list_date ≤ start_date` 过滤，消除尚未上市股票出现在回测池的问题。

---

## 5. Fama-French 因子重构

**涉及文件：** `core/factors/impl/fama_french.py`

- N+1 API 调用：`_get_market_capitalization` / `_get_pb_ratio` 从逐股 `for sym` 循环改为单次 `ak.stock_zh_a_daily_basic()` 全市场拉取（`_batch_daily_basic`）
- `ff_mkt` 值广播：从循环逐列赋值改为 `np.tile` 向量化
- 移除全局 `warnings.filterwarnings("ignore")`，改用局部 `import warnings`

---

## 6. 费率统一

**涉及文件：** `config/config.yaml`

| 项目 | 旧值 | 新值 | 对应费率 |
|------|------|------|---------|
| 印花税 | 0.001 (0.1%) | 0.0005 (0.05%) | 2023.8.28 起 |
| 过户费 | 0.00002 (0.002%) | 0.00001 (0.001%) | 2022 年起 |

与 `core/risk/fee.py` 默认值一致。

---

## 7. 因子列表统一

**涉及文件：**
- `core/factors/defaults.py` — 新增，导出 `DEFAULT_FACTOR_NAMES`
- `core/strategies/pipeline.py` — 原 `DEFAULT_FACTORS` 删除，改为导入 `DEFAULT_FACTOR_NAMES`
- `daily/2026-05-17/tuning_pipeline.py`
- `daily/2026-05-18/v1/emergency_experiment.py`
- `daily/2026-05-18/v1/param_scan.py`
- `daily/2026-05-23/ab_comparison.py`

---

## 8. 修复 bare `except: pass`

共 6 处，全部改为 `logger.warning` + 具体异常类型：

| 位置 | 行 | 场景 |
|------|-----|------|
| `live/signals/mss_dynamic.py` | 132/564/745/1008 | 文件删除、ST 查询、日期查询、symbol 名称查询 |
| `daily/2026-05-18/v2/live_signal.py` | 221 | ST 查询 |
| `scripts/generate_signal.py` | 79 | 日期解析 |

---

## 9. `print()` → `logger`

**涉及文件：**
- `core/factors/base/factor_hub.py` — 7 处 `print()` 改为 `logger.info/warning/error`
- `core/optimization/impl/attribution.py` — 添加 logger 定义
- `core/strategies/impl/config_signal_strategy.py` — 添加 logger 定义

---

## 10. gtja_ops 向量化

**涉及文件：** `core/factors/impl/gtja_ops.py`

| 函数 | 原实现 | 新实现 |
|------|--------|--------|
| `ts_rank` | `.rolling(window).apply(lambda s: s.iloc[-1])` | numpy 逐列逐窗口排名 |
| `ts_argmax` | `.rolling(window).apply(lambda s: s.argmax()+1)` | `_ts_arg_extreme` 向量化 |
| `ts_argmin` | 同上 | 同上 |
| `decay_linear` | `.rolling(window).apply(np.dot)` | numpy 循环 `np.dot` |
| `wma` | 同 decay_linear | 别名 → `decay_linear` |
| `_pow`/`_sqrt` | `.apply(lambda col: np.power/sqrt)` | 直接 `np.power`/`np.sqrt` |

---

## 11. mss_dynamic.py 拆分

原 `live/signals/mss_dynamic.py` 1452 行 → 5 个文件：

| 模块 | 行数 | 职责 |
|------|------|------|
| `mss_state.py` | 145 | DB 持久化、常量（FACTOR_NAMES, V7_ALLOCATION, STOP_LOSS_CONFIG 等） |
| `mss_factors.py` | 246 | 因子计算、择时信号、市场状态 |
| `mss_signal.py` | 453 | 订单生成、主流程、CLI |
| `mss_report.py` | 295 | HTML 报告、邮件发送 |
| `mss_dynamic.py` | 16 | 薄入口（导入 mss_signal.main） |

新 `mss_dynamic.py` 用法不变：
```bash
python3 -m live.signals.mss_dynamic --capital 50000
```

---

## 未在此次处理的问题

- **C1 SMTP 密码 git 历史清理** — `.env` 含 QQ 邮箱明文密码，需 `git filter-branch`
- **M1 测试覆盖** — Database/Pipeline/Live 覆盖率 0%（现有测试仅覆盖基础构造）
- **M3 BrokerAdapter** — 实盘下单接口 `IBrokerAdapter` 无任何实现类
- **M4 备用通知** — SMTP 发送失败只写文件，无人值守时管理员不知情
