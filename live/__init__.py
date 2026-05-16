"""Live Trading — 生产交易套件。

目录结构
---------
live/runner.py           核心调度器
live/signals/            信号生成(策略输出→调仓清单)
live/execution/          交易执行(券商API/订单管理)
live/storage/            持仓与成交存储
live/monitor/            绩效监控与日报

数据流
------
0830  pull:   IncrementalFetcher.fetch_all()  → 更新日线
0900  factor: FactorRunner.compute_all()      → 更新因子
0930  signal: generator.generate()            → 策略信号
            → combiner.combine()              → 多策略合并
            → order_manager.build_orders()    → 调仓清单
            → [人工/自动] 交易
1530  storage: positions.save_snapshot()       → 持仓快照
1600  monitor: dashboard.generate_report()     → 日绩效报告
"""
from __future__ import annotations
