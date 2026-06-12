"""回测版本追踪 — 将最新的回测结果追加到追踪日志。

用法: python3 daily/track_backtest.py
       python3 daily/track_backtest.py --read   # 只读最近记录
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACK_FILE = "daily/backtest_log.md"
RESULTS_DIR = "daily/2026-06-02/results"


def load_results():
    path = os.path.join(RESULTS_DIR, "full_backtest.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_experiments():
    path = os.path.join(RESULTS_DIR, "experiments.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def append_log():
    data = load_results()
    if not data:
        print("❌ 未找到 full_backtest.json")
        return

    meta = data.get("meta", {})
    main = data.get("main_strategy", {})

    today = datetime.now().strftime("%Y-%m-%d")
    line = f"| {today} | V6 | {meta.get('data_end','?')} | 1796 | 5515 |"

    for lbl in ["实盘口径_t5", "实盘口径_t3", "研究口径_t5", "研究口径_t3"]:
        if lbl in main:
            m = main[lbl]["metrics"]
            line += f" {m['annual_return']*100:.2f}% | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f}% | {m['calmar']:.3f} | {m['win_rate']*100:.1f}% |"

    os.makedirs(os.path.dirname(TRACK_FILE), exist_ok=True)

    if not os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "w") as f:
            f.write("# ZEquant 回测版本追踪\n\n")
            f.write("| 日期 | 版本 | 数据截止 | 天数 | 标的数 |")
            f.write(" 口径 | 年化% | Sharpe | 回撤% | Calmar | 胜率 |")
            f.write("\n")
            f.write("|:---:|:---:|:-------:|:---:|:----:|")
            f.write(":----:|:----:|:-----:|:----:|:----:|:---:|")
            f.write("\n")

    with open(TRACK_FILE, "a") as f:
        cols = line.split("|")
        if len(meta) > 0:
            f.write(f"| {today} | V6 | {meta.get('data_end','?')} | {meta.get('n_days','?')} | {meta.get('n_stocks','?')} |\n")
            for lbl in ["实盘口径_t5", "实盘口径_t3", "研究口径_t5", "研究口径_t3"]:
                if lbl in main:
                    m = main[lbl]["metrics"]
                    f.write(f"| | | | | | {lbl} | {m['annual_return']*100:.2f}% | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f}% | {m['calmar']:.3f} | {m['win_rate']*100:.1f}% |\n")

    # 实验记录
    exp = load_experiments()
    if exp:
        f.write(f"| {today} | 实验 | 消融实验 | | | | | | | | |\n")
        for r in exp:
            m = r["metrics"]
            f.write(f"| | {r['name']} | | | | | {m['annual_return']*100:.2f}% | {m['sharpe']:.3f} | {abs(m['max_drawdown'])*100:.2f}% | {m['calmar']:.3f} | {m['win_rate']*100:.1f}% |\n")
        f.write("\n")

    print(f"✅ 已追加到 {TRACK_FILE}")


def read_log():
    if not os.path.exists(TRACK_FILE):
        print("❌ 追踪日志不存在")
        return
    with open(TRACK_FILE) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="回测追踪")
    parser.add_argument("--read", action="store_true", help="只读")
    args = parser.parse_args()
    if args.read:
        read_log()
    else:
        append_log()


if __name__ == "__main__":
    main()
