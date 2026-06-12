"""IR时序衰减诊断 — 各因子IR随时间变化

用法:
    python3 daily/2026-06-12/ir_timeseries.py
    python3 daily/2026-06-12/ir_timeseries.py --factors gtja142 gtja144 a41
    python3 daily/2026-06-12/ir_timeseries.py --window 30
    python3 daily/2026-06-12/ir_timeseries.py --output ir_data.json

输出:
    - 终端: 各因子IR时序摘要表
    - JSON: 各因子每窗口IC/IR数据(可用于绘图)
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb

DB_PATH = os.path.abspath("./data/quant_data.db")

# 实盘权重因子(20个监控因子)
MONITORED_FACTORS = [
    'gtja142', 'gtja168', 'gtja49', 'gtja117', 'a41', 'gtja141', 'gtja144',
    'a42', 'gtja12', 'a80', 'gtja123', 'gtja83', 'a85', 'gtja104', 'gtja62',
    'volatility_20', 'a69', 'gtja13', 'gtja113', 'gtja34'
]


def load_factor_data(start_date="2020-01-01", end_date="2026-06-11"):
    """加载因子和行情数据"""
    t0 = time.time()
    conn = duckdb.connect(DB_PATH, read_only=True)
    all_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'"
    ).fetchall()]
    factors = [c for c in MONITORED_FACTORS if c in all_cols]
    factor_cols = ", ".join([f'f."{c}"' for c in factors])
    df = conn.execute(
        f'SELECT f.date,f.symbol,b.close,{factor_cols} '
        f'FROM factors_wide f LEFT JOIN daily_bars b ON f.date=b.date AND f.symbol=b.symbol '
        f"WHERE f.date>='{start_date}' AND f.date<='{end_date}' ORDER BY f.date,f.symbol"
    ).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    conn.close()
    print(f"[INFO] 数据加载: {len(df)}行, {len(factors)}因子 ({time.time()-t0:.1f}s)", file=sys.stderr)
    return df, factors


def calc_ic_series(factor_data, factor_name, forward_days=10):
    """计算单因子IC时序(滚动窗口)"""
    df = factor_data[['date', 'symbol', factor_name, 'close']].copy()
    df = df.sort_values(['symbol', 'date'])
    # 前瞻收益: T日因子值 → T+N日收益
    df['close_future'] = df.groupby('symbol')['close'].shift(-forward_days)
    df['fwd_ret'] = df['close_future'] / df['close'] - 1
    df = df.dropna(subset=['fwd_ret', factor_name])
    # 按日截面计算Spearman IC
    ic_list = []
    for date, group in df.groupby('date'):
        if len(group) < 10:
            continue
        ic = group[factor_name].corr(group['fwd_ret'], method='spearman')
        ic_list.append({'date': date, 'ic': ic})
    return pd.DataFrame(ic_list)


def calc_rolling_ir(ic_df, window=60):
    """计算滚动IR = mean(IC) / std(IC)"""
    if len(ic_df) < window:
        return pd.DataFrame()
    ic_df = ic_df.sort_values('date').reset_index(drop=True)
    records = []
    for i in range(window, len(ic_df) + 1):
        window_ics = ic_df['ic'].iloc[i - window:i]
        mean_ic = window_ics.mean()
        std_ic = window_ics.std()
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0
        records.append({
            'date': ic_df['date'].iloc[i - 1],
            'ir': ir,
            'ic_mean': mean_ic,
            'ic_std': std_ic,
            'n_days': len(window_ics),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="IR时序衰减诊断")
    parser.add_argument("--factors", nargs="*", default=MONITORED_FACTORS, help="因子列表")
    parser.add_argument("--start", default="2020-01-01", help="数据起始日期")
    parser.add_argument("--end", default="2026-06-11", help="数据结束日期")
    parser.add_argument("--forward", type=int, default=10, help="前瞻天数")
    parser.add_argument("--window", type=int, default=60, help="IR滚动窗口(天)")
    parser.add_argument("--output", default=None, help="输出JSON文件路径")
    args = parser.parse_args()

    # 加载数据
    df, available_factors = load_factor_data(args.start, args.end)
    target_factors = [f for f in args.factors if f in available_factors]

    # 衰减分类
    DEAD_FACTORS = {"gtja142", "gtja141"}
    REVERSAL_FACTORS = {"gtja144", "a41", "gtja117", "gtja49", "gtja13", "gtja34", "a69", "gtja113"}
    DECAY_FACTORS = {"a80", "volatility_20"}

    all_results = {}
    summary_rows = []

    print("\n" + "=" * 90)
    print(f"IR时序衰减诊断 (窗口={args.window}天, 前瞻={args.forward}天)")
    print("=" * 90)

    for factor_name in target_factors:
        t0 = time.time()
        ic_df = calc_ic_series(df, factor_name, forward_days=args.forward)
        if len(ic_df) < args.window:
            print(f"  {factor_name}: IC数据不足({len(ic_df)}天), 跳过", file=sys.stderr)
            continue

        ir_df = calc_rolling_ir(ic_df, window=args.window)
        if len(ir_df) == 0:
            continue

        # 计算衰减指标
        historical_ir = ir_df['ir'].iloc[:len(ir_df) // 2].mean() if len(ir_df) > 10 else ir_df['ir'].iloc[0]
        recent_ir = ir_df['ir'].iloc[-10:].mean()
        latest_ir = ir_df['ir'].iloc[-1]
        decay_ratio = recent_ir / historical_ir if abs(historical_ir) > 1e-6 else 0
        ic_direction_flipped = (historical_ir * recent_ir < 0)

        # 分类
        if factor_name in DEAD_FACTORS:
            category = "DEAD"
        elif factor_name in REVERSAL_FACTORS:
            category = "REVERSAL"
        elif factor_name in DECAY_FACTORS:
            category = "DECAY"
        else:
            category = "OK"

        # 趋势: 最近30天IR的斜率
        if len(ir_df) >= 30:
            recent_30 = ir_df['ir'].iloc[-30:].values
            slope = np.polyfit(np.arange(len(recent_30)), recent_30, 1)[0]
        else:
            slope = 0

        summary_rows.append({
            'factor': factor_name,
            'category': category,
            'historical_ir': round(historical_ir, 4),
            'recent_ir': round(recent_ir, 4),
            'latest_ir': round(latest_ir, 4),
            'decay_ratio': round(decay_ratio, 4),
            'ic_flipped': ic_direction_flipped,
            'trend_slope': round(slope, 6),
        })

        # 保存IR时序数据
        all_results[factor_name] = {
            'category': category,
            'historical_ir': round(float(historical_ir), 4),
            'recent_ir': round(float(recent_ir), 4),
            'decay_ratio': round(float(decay_ratio), 4),
            'ic_flipped': bool(ic_direction_flipped),
            'ir_series': [
                {'date': str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date']),
                 'ir': round(float(row['ir']), 4)}
                for _, row in ir_df.iterrows()
            ],
        }

        elapsed = time.time() - t0
        flip_mark = "⚠FLIP" if ic_direction_flipped else ""
        print(f"  {factor_name:<15} [{category:<8}] "
              f"历史IR={historical_ir:>7.3f} 近期IR={recent_ir:>7.3f} "
              f"衰减比={decay_ratio:>7.2f} 斜率={slope:>9.6f} {flip_mark} ({elapsed:.1f}s)")

    # 汇总表
    print("\n" + "=" * 90)
    print("IR衰减汇总")
    print("=" * 90)
    print(f"{'因子':<15} {'分类':<10} {'历史IR':>8} {'近期IR':>8} {'衰减比':>8} {'方向反转':>8} {'趋势':>10}")
    print("-" * 70)
    for r in sorted(summary_rows, key=lambda x: x['decay_ratio']):
        flip = "是" if r['ic_flipped'] else "否"
        trend = "↑" if r['trend_slope'] > 0.001 else "↓" if r['trend_slope'] < -0.001 else "→"
        print(f"{r['factor']:<15} {r['category']:<10} {r['historical_ir']:>8.3f} "
              f"{r['recent_ir']:>8.3f} {r['decay_ratio']:>8.2f} {flip:>8} {r['trend_slope']:>10.6f}{trend}")

    # 保存JSON
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ir_timeseries.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nIR时序数据已保存到 {output_path}")


if __name__ == "__main__":
    main()