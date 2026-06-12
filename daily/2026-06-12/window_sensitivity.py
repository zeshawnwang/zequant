"""窗口敏感性分析 — 不同lookback窗口对IR衰减判断的影响

用法:
    python3 daily/2026-06-12/window_sensitivity.py
    python3 daily/2026-06-12/window_sensitivity.py --windows 30 60 90 120 180
    python3 daily/2026-06-12/window_sensitivity.py --factors gtja142 gtja144

输出:
    - 终端: 各窗口下各因子的IR和衰减分类
    - JSON: 窗口敏感性数据
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb

DB_PATH = os.path.abspath("./data/quant_data.db")

MONITORED_FACTORS = [
    'gtja142', 'gtja168', 'gtja49', 'gtja117', 'a41', 'gtja141', 'gtja144',
    'a42', 'gtja12', 'a80', 'gtja123', 'gtja83', 'a85', 'gtja104', 'gtja62',
    'volatility_20', 'a69', 'gtja13', 'gtja113', 'gtja34'
]


def load_factor_data(start_date="2020-01-01", end_date="2026-06-11"):
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
    return df, factors


def calc_ic_series(factor_data, factor_name, forward_days=10):
    df = factor_data[['date', 'symbol', factor_name, 'close']].copy()
    df = df.sort_values(['symbol', 'date'])
    df['close_future'] = df.groupby('symbol')['close'].shift(-forward_days)
    df['fwd_ret'] = df['close_future'] / df['close'] - 1
    df = df.dropna(subset=['fwd_ret', factor_name])
    ic_list = []
    for date, group in df.groupby('date'):
        if len(group) < 10:
            continue
        ic = group[factor_name].corr(group['fwd_ret'], method='spearman')
        ic_list.append({'date': date, 'ic': ic})
    return pd.DataFrame(ic_list)


def calc_rolling_ir(ic_df, window=60):
    if len(ic_df) < window:
        return pd.DataFrame()
    ic_df = ic_df.sort_values('date').reset_index(drop=True)
    records = []
    for i in range(window, len(ic_df) + 1):
        window_ics = ic_df['ic'].iloc[i - window:i]
        mean_ic = window_ics.mean()
        std_ic = window_ics.std()
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0
        records.append({'date': ic_df['date'].iloc[i - 1], 'ir': ir, 'ic_mean': mean_ic})
    return pd.DataFrame(records)


def classify_factor(historical_ir, recent_ir, decay_threshold=0.5, reversal_check=True):
    """分类因子状态"""
    if abs(historical_ir) < 0.01:
        return "NEUTRAL"
    decay_ratio = abs(recent_ir / historical_ir) if abs(historical_ir) > 1e-6 else 0
    ic_flipped = (historical_ir * recent_ir < 0) and reversal_check
    if ic_flipped:
        return "REVERSAL"
    if decay_ratio < 0.2:
        return "DEAD"
    if decay_ratio < 0.5:
        return "DECAY"
    return "OK"


def main():
    parser = argparse.ArgumentParser(description="窗口敏感性分析")
    parser.add_argument("--factors", nargs="*", default=MONITORED_FACTORS, help="因子列表")
    parser.add_argument("--start", default="2020-01-01", help="数据起始日期")
    parser.add_argument("--end", default="2026-06-11", help="数据结束日期")
    parser.add_argument("--forward", type=int, default=10, help="前瞻天数")
    parser.add_argument("--windows", nargs="*", type=int, default=[30, 60, 90, 120, 180],
                        help="IR滚动窗口列表(天)")
    parser.add_argument("--output", default=None, help="输出JSON文件路径")
    args = parser.parse_args()

    df, available_factors = load_factor_data(args.start, args.end)
    target_factors = [f for f in args.factors if f in available_factors]

    all_results = {}

    print("\n" + "=" * 100)
    print(f"窗口敏感性分析 (前瞻={args.forward}天)")
    print("=" * 100)

    for factor_name in target_factors:
        t0 = time.time()
        ic_df = calc_ic_series(df, factor_name, forward_days=args.forward)
        if len(ic_df) < max(args.windows):
            print(f"  {factor_name}: IC数据不足({len(ic_df)}天), 跳过", file=sys.stderr)
            continue

        factor_result = {"windows": {}}
        print(f"\n  {factor_name}:")

        for window in args.windows:
            ir_df = calc_rolling_ir(ic_df, window=window)
            if len(ir_df) < 10:
                continue

            historical_ir = ir_df['ir'].iloc[:len(ir_df) // 2].mean()
            recent_ir = ir_df['ir'].iloc[-10:].mean()
            latest_ir = ir_df['ir'].iloc[-1]
            category = classify_factor(historical_ir, recent_ir)
            decay_ratio = recent_ir / historical_ir if abs(historical_ir) > 1e-6 else 0

            factor_result["windows"][str(window)] = {
                "historical_ir": round(historical_ir, 4),
                "recent_ir": round(recent_ir, 4),
                "latest_ir": round(latest_ir, 4),
                "decay_ratio": round(decay_ratio, 4),
                "category": category,
            }

            print(f"    W={window:>3}d: 历史IR={historical_ir:>7.3f} 近期IR={recent_ir:>7.3f} "
                  f"衰减比={decay_ratio:>7.2f} → {category}")

        # 判断窗口敏感性
        categories = [w_data["category"] for w_data in factor_result["windows"].values()]
        unique_cats = set(categories)
        if len(unique_cats) == 1:
            sensitivity = "STABLE"
        elif len(unique_cats) <= 2:
            sensitivity = "MODERATE"
        else:
            sensitivity = "SENSITIVE"
        factor_result["sensitivity"] = sensitivity
        print(f"    → 窗口敏感性: {sensitivity} (分类: {', '.join(unique_cats)})")

        all_results[factor_name] = factor_result
        elapsed = time.time() - t0
        print(f"    ({elapsed:.1f}s)")

    # 汇总
    print("\n" + "=" * 100)
    print("窗口敏感性汇总")
    print("=" * 100)
    print(f"{'因子':<15} {'敏感性':<12} ", end="")
    for w in args.windows:
        print(f"{'W'+str(w):>10}", end="")
    print()
    print("-" * (15 + 12 + 10 * len(args.windows)))

    for factor_name, data in all_results.items():
        print(f"{factor_name:<15} {data['sensitivity']:<12} ", end="")
        for w in args.windows:
            w_data = data["windows"].get(str(w), {})
            cat = w_data.get("category", "N/A")
            print(f"{cat:>10}", end="")
        print()

    # 保存JSON
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "window_sensitivity.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n窗口敏感性数据已保存到 {output_path}")


if __name__ == "__main__":
    main()