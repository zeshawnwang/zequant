# ZEquant 项目 Code Review 报告

> 审查日期: 2026-06-10
> 审查范围: 全项目代码（core/、live/、scripts/、tests/、config/）

---

## 一、总体评价

**项目质量: B+ (良好)**

ZEquant 是一个成熟的 A 股量化研究框架，架构清晰、功能完整。项目具备良好的分层设计（数据源 → 因子 → 选股 → 择时 → 仓位 → 回测 → 实盘），DuckDB 单文件数据库设计简洁高效，27 个策略沉淀完整且有 walk-forward 验证。以下从多个维度给出详细评审。

---

## 二、架构设计 (A-)

### 优点
1. **可插拔三段式架构**: 选股/择时/仓位解耦良好，组合灵活
2. **双评估路径**: Pipeline（向量化快速扫描）+ BacktestEngine（事件驱动完整回测），覆盖研究和生产场景
3. **FactorHub 注册中心**: 因子声明式注册，纯函数设计易测试
4. **MarketStateSelector 动态切换**: 根据市场状态切换子策略，实用且验证充分
5. **DuckDB 宽表设计**: 列存对多因子场景友好，避免 pivot 开销

### 需改进
1. **代码重复**: `get_price_limit_pct()` 在 `core/screening/universe.py` 和 `core/strategies/pipeline.py` 中重复定义
2. **代码重复**: `FUND_PREFIXES` 和 `is_fund_symbol()` 在 `live/data_updater.py` 和 `core/datasource/sources/akshare_source.py` 中重复定义
3. **模块耦合**: `pipeline.py` 单文件 730+ 行，承载数据加载/信号构建/回测/指标/窗口分析/导出，职责过重，建议拆分

---

## 三、代码质量 (B+)

### 优点
1. **文档注释完善**: 每个模块有中文 docstring，解释设计要点和用法
2. **类型标注**: 核心 API 有类型注解（typing）
3. **Pydantic 配置校验**: `config_model.py` 用 Pydantic v2 做启动时类型检查，防止运行时配置错误
4. **兼容性考虑**: `Database` 同时支持宽表/长表 API，向后兼容
5. **无 TODO/FIXME**: 代码库中没有遗留的临时标记

### 需改进
1. **bare except（严重）**: 存在 4 处 bare `except:` 未指定异常类型（`live/signals/mss_signal.py:125,320`、`mss_factors.py:194`、`mss_state.py:116`），会吞掉所有异常包括 `KeyboardInterrupt`
2. **过宽异常捕获**: `core/database.py` 中多处 `except Exception: return None` 静默吞错，可能掩盖数据库连接问题
3. **f-string 日志**: 部分 logger 使用 f-string（如 `pipeline.py:251,549`），应使用 `%s` 占位符避免不必要的字符串格式化
4. **魔数**: Pipeline 回测中的 `1e-10`、`-1e10`、`0.2`(buf) 等未抽常量
5. **`iterrows()` 性能**: `pipeline.py:316` 对全市场 daily_bars 使用 `iterrows()` 遍历，大数据集下性能极差，应向量化

---

## 四、安全性 (C+)

### 严重问题
1. **SMTP 密码明文写入 .env**: `.env` 文件中 `SMTP_PASS=mxijaqrhgioebajc` 是明文应用专用密码。虽然 `.gitignore` 已排除 `.env`，但：
   - 工作区文件仍然存在风险
   - 如果曾经不在 `.gitignore` 中提交过，git 历史中仍会存在
   - 建议使用系统密钥链或环境变量注入

### 中等问题
2. **SQL 注入风险低但存在**: `Database.get_max_date()` 中 `table` 和 `column` 参数直接拼入 SQL，然是内部调用，仍建议白名单校验
3. **自定义 dotenv 解析器**: `live/notification/__init__.py` 手写 `.env` 解析而非使用 `python-dotenv`（已在 requirements.txt 中），容易出现边情况

---

## 五、测试质量 (B-)

### 优点
1. 31 个单元测试覆盖核心路径（回测引擎、策略、选股器、因子Hub）
2. 测试用例使用固定 seed 确保可复现

### 需改进
1. **覆盖率不足**: 未覆盖的关键模块:
   - `live/` 实盘模块无测试
   - `core/datasource/` 数据源无测试
   - `core/strategies/pipeline.py` 完整 Pipeline 无测试
   - `core/risk/fee.py` FeeCalculator 无测试
2. **无集成测试**: 缺少端到端流程测试（数据→因子→回测→信号）
3. **无 mock**: 数据源测试应 mock 外部 API（akshare/baostock）

---

## 六、业务逻辑问题 (B)

### 需修复
1. **过户费计算错误（中等）**: `core/risk/fee.py:44,59` 中 `transfer_fee = ... if symbol.startswith('6') else 0`，注释写"沪深统一"（2022年起），代码只对6开头（沪市）收取，深市股票（0开头）被错误免除
2. **FeeCalculator 配置不一致**: `config.yaml` 中 `transfer_fee: 0.00001` (0.001%)，而注释说"过户费 0.001%"，数值正确但与代码行为矛盾（只收沪市）

### 需关注
3. **前向收益计算**: `pipeline.py:216-218` 使用 `cl[d+1]` 作为前向收益（次日收盘），实际应为次日可成交价格（开盘价），在价格滑点偏差
4. **Universe 过滤时序**: pipeline 中 `_build_universe_mask` 在停牌/涨跌停检查时用 `iterrows()`，在 5000+ 股票 × 1500+ 天的场景下极慢

---

## 七、工程实践 (B)

### 优点
1. **`.gitignore` 完善**: 数据库、日志、venv、IDE 文件全部排除
2. **多环境支持**: config.yaml + .env 分离敏感信息
3. **增量更新**: 数据源支持增量拉取，避免重复下载
4. **多源兜底**: akshare → baostock → efinance 自动降级

### 需改进
1. **`sys.path.insert` 滥用**: 全项目有 46 处手动操作 `sys.path`，说明包结构不够规范。应使用 `pyproject.toml` + `-e` 安装或统的项目根包
2. **无 CI/CD**: 没有 GitHub Actions / 自动化测试流水线
3. **无 linter/formatter 配置**: 缺少 `ruff.toml` / `pyproject.toml [tool.ruff]`，代码风格靠人工维护
4. **依赖版本下限**: `requirements.txt` 只指定最低版本（`>=`），生产环境建议锁定（`==` 或 lockfile）

---

## 八、性能 (B+)

### 优点
1. **NumPy 向量化**: Pipeline 回测核心循环高效
2. **DuckDB 配置自适应**: 自动检测 CPU/内存，合理设置 threads 和 memory_limit
3. **批量写入**: `save_factors` 使用事务 + batch 写入

### 需改进
1. **`iterrows()` 热路径**: `pipeline.py:316` universe 过滤对全量 bars 用 `iterrows()`，建议用 merge + 向量化逻辑
2. **Z-score 循环**: `pipeline.py:221-230` 逐天逐因子循环做 zscore，可用 `np.apply_along_axis` 或全量向量化
3. **内存峰值**: `pipeline.load()` 一次性加载全市场所有因子为 3D array，5000股×1500天×60因子 ≈ 1.7 GB float32

---

## 九、优先级排序的改进建议

| 优先级 | 类别 | 建议 |
|:---:|:---:|:---|
| P0 | 安全 | 确认 `.env` 从未被提交到 git 历史；考虑用 keychain/vault |
| P0 | Bug | 修复过户费只收沪市的逻辑错误（深市同样应收取） |
| P1 | 质量 | 消除 4 处 bare `except:`，改为 `except Exception` |
| P1 | 工程 | 引入 `pyproject.toml` + `pip install -e .`，消除 sys.path hack |
| P1 | 性能 | `_build_universe_mask` 中 iterrows → 向量化 |
| P2 | 测试 | 补充 live/ 和 datasource/ 的单元测试 |
| P2 | 质量 | 抽取 `get_price_limit_pct` / `FUND_PREFIXES` 到共享模块消除重复 |
| P2 | 工程 | 添加 ruff linter + CI 自动化 |
| P3 | 架构 | 拆分 `pipeline.py` 为 data_loader / backtester / metrics 子模块 |
| P3 | 工程 | requirements.txt → pyproject.toml + 锁定版本 |

---

## 十、总结

ZEquant 是一个**功能完整、架构合理**的量化研究框架，策略体系经过充分的回测和 walk-forward 验证。主要风险在于：

1. **生产安全**: SMTP 密码管理和过户费计算错误需立即修复
2. **工程成熟度**: 缺少 CI、linter、测试覆盖率监控，长期维护有隐患
3. **性能瓶颈**: Universe 过滤的 iterrows 在大数据集下是主要热路径

建议按 P0→P1→P2 优先级逐步改进，整体代码质量良好，无需大规模重构。