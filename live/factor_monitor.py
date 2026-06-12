"""因子衰减监控 — 实盘入口。

定期（建议每月）运行，检测实盘使用的因子是否在衰减。
输出预警报告到 data_live/factor_monitor/ 目录。

用法:
    python3 -m live.factor_monitor                     # 监控实盘56因子
    python3 -m live.factor_monitor --top 20            # 只监控权重前20因子
    python3 -m live.factor_monitor --end 2026-06-01    # 指定截止日期
    python3 -m live.factor_monitor --window 60         # 60天滚动窗口(更敏感)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date

# 实盘因子列表 — 从 mss_state 中导入
from live.signals.mss_state import FACTOR_NAMES

# 实盘使用的多因子权重 (权重绝对值前20大)
LIVE_FACTOR_WEIGHTS = {
    'gtja142': 0.3005, 'gtja168': 0.3003, 'gtja117': 0.2324,
    'gtja144': 0.2045, 'a41': 0.2152, 'a42': 0.1999,
    'gtja141': 0.2104, 'gtja49': -0.2478, 'gtja12': -0.1859,
    'a80': 0.1689, 'gtja123': 0.1666, 'a85': 0.1419,
    'gtja83': 0.1429, 'volatility_20': -0.1127, 'gtja104': -0.1303,
    'gtja62': 0.1181, 'gtja113': -0.0874, 'a69': -0.0961,
    'gtja13': 0.0903, 'gtja34': -0.0816,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='因子衰减监控')
    parser.add_argument('--end', type=str, default=None,
                        help='监控截止日期 (默认: 数据库最新日期)')
    parser.add_argument('--top', type=int, default=0,
                        help='只监控权重前N大因子 (默认: 全部实盘因子)')
    parser.add_argument('--window', type=int, default=120,
                        help='滚动窗口天数 (默认120)')
    parser.add_argument('--step', type=int, default=20,
                        help='滑动步长天数 (默认20)')
    parser.add_argument('--lookback', type=float, default=3.0,
                        help='往前看几年 (默认3)')
    parser.add_argument('--json', action='store_true',
                        help='输出JSON格式报告')
    args = parser.parse_args()

    from core.database import Database

    db = Database("./data/quant_data.db")

    # 确定监控因子列表
    if args.top > 0:
        sorted_factors = sorted(
            LIVE_FACTOR_WEIGHTS.items(), key=lambda x: abs(x[1]), reverse=True
        )
        factor_names = [f for f, _ in sorted_factors[:args.top]]
        logger.info(f"监控权重前 {args.top} 因子")
    else:
        # 排除 close/returns 等非因子列
        factor_names = [f for f in FACTOR_NAMES
                        if f not in ('close', 'returns', 'volume_ratio')]
        logger.info(f"监控全部 {len(factor_names)} 实盘因子")

    # 确定截止日期
    end_date = args.end
    if not end_date:
        row = db.conn.execute("SELECT MAX(date) FROM factors_wide").fetchone()
        end_date = str(row[0]) if row and row[0] else str(date.today())
    logger.info(f"截止日期: {end_date}")

    # 执行监控
    from core.research.impl.factor_monitor import FactorDecayMonitor

    monitor = FactorDecayMonitor(
        db, window_days=args.window, step_days=args.step
    )
    report = monitor.run(
        factor_names=factor_names,
        end_date=end_date,
        lookback_years=args.lookback,
    )

    # 输出
    print(report.summary())

    # 写入文件
    out_dir = "data_live/factor_monitor"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"report_{end_date.replace('-', '')}.json")

    report_data = {
        "run_date": report.run_date,
        "end_date": report.end_date,
        "n_factors": report.n_factors,
        "n_windows": report.n_windows,
        "window_days": report.window_days,
        "alerts": [
            {"factor": a.factor_name, "type": a.alert_type,
             "severity": a.severity, "message": a.message,
             "current_ir": round(a.current_ir, 4),
             "historical_ir": round(a.historical_ir, 4),
             "decay_ratio": round(a.decay_ratio, 4)}
            for a in report.alerts
        ],
        "health_scores": report.health_scores,
    }
    with open(out_file, 'w') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"报告已写入 {out_file}")

    db.close()


if __name__ == "__main__":
    main()