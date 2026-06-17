# ZEquant Daily 实验规范

> 本规范定义 `daily/` 目录下所有实验的文件组织、命名、记录要求。
> 每一位研究者（含AI）必须严格遵守。

---

## 1. 目录结构

```
daily/
├── daily_convention.md        # 本规范文件
├── daily_research_log.md      # 研究总记录（跨日期索引）
├── 2026-06-12/                # 按日期组织
│   ├── README.md              # 当日研究概述（背景、目标、结论）
│   ├── .autoresearch.md       # autoresearch 优化配置（如使用）
│   ├── evaluate.py            # 评估脚本
│   ├── ablation.py            # 消融实验脚本
│   ├── results.tsv            # 实验结果记录
│   ├── weights/               # 权重文件子目录
│   │   ├── A0.json
│   │   └── ...
│   └── *.log                  # 运行日志（如有）
└── 2026-06-15/
    ├── README.md
    ├── ...
```

## 2. 文件规范

### 2.1 必须存在的文件

| 文件 | 说明 | 格式 |
|:-----|:-----|:-----|
| `README.md` | 当日研究概述 | Markdown |
| `results.tsv` | 实验结果记录 | TSV（Tab分隔） |

### 2.2 脚本文件

- 每个脚本文件顶部必须有 docstring，说明：目的、输入、输出、用法
- 脚本输出的关键指标必须打印到 stdout 最后一行（供 autoresearch 解析）
- 脚本必须可通过 `python daily/YYYY-MM-DD/script.py` 从项目根目录运行

### 2.3 数据文件

- JSON 权重文件放 `weights/` 子目录
- 大型结果文件放 `results/` 子目录
- 日志文件以 `.log` 结尾
- 临时文件以 `_` 前缀或放 `_temp/` 子目录，实验完成后删除

### 2.4 命名规范

- 脚本：`snake_case.py`，名称反映功能（如 `ablation.py`, `ir_timeseries.py`）
- 权重：`大写字母+数字.json`（如 `A0.json`, `A4.json`, `ga_optimized.json`）
- 结果：`results.tsv`（统一结果表）或 `具体名称.json`（详细结果）

## 3. results.tsv 格式

```
时间戳\t改动摘要\t指标1\t指标2\t...\tkept|discarded
```

- 第一行为表头
- 每次实验追加一行
- `kept` 表示保留改动，`discarded` 表示丢弃

## 4. README.md 模板

```markdown
# YYYY-MM-DD 研究标题

## 背景
[为什么做这个研究，从什么问题出发]

## 目标
[要回答什么问题，验证什么假设]

## 方法
[用了什么方法、脚本、数据范围]

## 结果摘要
[关键发现，指向 results.tsv 的具体行]

## 结论与后续
[结论和下一步行动]
```

## 5. daily_research_log.md 规范

每条记录包含：

| 字段 | 说明 |
|:-----|:-----|
| 日期 | YYYY-MM-DD |
| 标题 | 研究标题 |
| 背景 | 从什么问题出发 |
| 范围 | 涉及的策略/模块/数据 |
| 方式 | 实验方法 |
| 关键结果 | 核心数字和发现 |
| 结论 | 可操作的结论 |
| 影响范围 | 对实盘/其他策略的影响 |

## 6. 禁止事项

- ❌ 不在日期目录外放实验脚本或结果
- ❌ 不提交 `results.tsv` 到 git（临时实验数据）
- ❌ 不删除历史日期目录（即使实验失败也保留记录）
- ❌ 不在脚本中硬编码绝对路径
- ❌ 不遗漏 results.tsv 记录（每次实验必须追加）

---

*最后更新：2026-06-15*