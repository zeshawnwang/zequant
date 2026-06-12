"""因子衰减权重消融实验 — A0基线 ~ A5健康度加权

用法:
    python3 daily/2026-06-12/ablation.py          # 运行全部实验
    python3 daily/2026-06-12/ablation.py A0 A1    # 只运行指定实验
    python3 daily/2026-06-12/ablation.py --all-details  # 输出每组详细指标

实验组:
    A0: 基线 — 当前实盘权重
    A1: 降dead — gtja142/gtja141权重降为0
    A2: 剔除reversal — 8个reversal因子权重降为0
    A3: 合并 — dead+reversal全降为0
    A4: 翻转reversal — 8个reversal因子权重取反
    A5: 健康度加权 — 权重乘以健康度评分
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 当前实盘权重(从config.json提取)
BASELINE_WEIGHTS = {
    "ff_mkt": 0.0413, "gtja142": 0.3005, "gtja144": 0.2045, "gtja171": -0.0443,
    "gtja103": -0.0147, "gtja85": -0.0258, "a88": -0.035, "a31": -0.0249,
    "rsi_14": 0.0253, "gtja139": -0.0112, "gtja123": 0.1666, "a42": 0.1999,
    "a41": 0.2152, "a97": -0.0734, "gtja148": -0.0048, "gtja99": -0.059,
    "gtja117": 0.2324, "gtja76": 0.0032, "gtja90": 0.0437, "volatility_20": -0.1127,
    "gtja113": -0.0874, "gtja141": 0.2104, "a99": -0.072, "gtja12": -0.1859,
    "gtja83": 0.1429, "gtja164": 0.0235, "a98": 0.0657, "gtja49": -0.2478,
    "gtja121": -0.0095, "a85": 0.1419, "gtja104": -0.1303, "gtja185": -0.0565,
    "gtja176": -0.075, "a80": 0.1689, "gtja62": 0.1181, "a8": 0.0657,
    "gtja34": -0.0816, "returns": -0.0508, "gtja168": 0.3003, "gtja108": -0.0791,
    "gtja105": 0.0686, "gtja127": -0.0506, "a27": -0.0627, "a64": 0.0874,
    "gtja91": -0.0399, "a30": -0.0666, "a69": -0.0961, "a91": -0.0582,
    "gtja13": 0.0903, "gtja120": 0.055
}

# 因子衰减监控报告中的分类
DEAD_FACTORS = {"gtja142", "gtja141"}
REVERSAL_FACTORS = {"gtja144", "a41", "gtja117", "gtja49", "gtja13", "gtja34", "a69", "gtja113"}
DECAY_FACTORS = {"a80", "volatility_20"}

# 健康度评分(从report_20260611.json)
HEALTH_SCORES = {
    "gtja142": 0.5156, "gtja168": 0.6376, "gtja49": 0.5085, "gtja117": 0.6691,
    "a41": 0.5313, "gtja141": 0.4692, "gtja144": 0.5313, "a42": 0.6421,
    "gtja12": 0.5313, "a80": 0.6003, "gtja123": 0.7236, "gtja83": 0.8386,
    "a85": 0.5815, "gtja104": 0.3972, "gtja62": 0.7248, "volatility_20": 0.4503,
    "a69": 0.4754, "gtja13": 0.5313, "gtja113": 0.5198, "gtja34": 0.5965,
}


def normalize_weights(w):
    """权重绝对值之和归一化"""
    s = sum(abs(v) for v in w.values())
    if s > 0:
        return {k: v / s for k, v in w.items()}
    return w


def gen_a0_baseline():
    """A0: 当前实盘权重(基线)"""
    return normalize_weights(dict(BASELINE_WEIGHTS))


def gen_a1_zero_dead():
    """A1: dead因子权重降为0"""
    w = dict(BASELINE_WEIGHTS)
    for f in DEAD_FACTORS:
        if f in w:
            w[f] = 0.0
    return normalize_weights(w)


def gen_a2_zero_reversal():
    """A2: reversal因子权重降为0"""
    w = dict(BASELINE_WEIGHTS)
    for f in REVERSAL_FACTORS:
        if f in w:
            w[f] = 0.0
    return normalize_weights(w)


def gen_a3_zero_all_alert():
    """A3: dead+reversal全降为0"""
    w = dict(BASELINE_WEIGHTS)
    for f in DEAD_FACTORS | REVERSAL_FACTORS:
        if f in w:
            w[f] = 0.0
    return normalize_weights(w)


def gen_a4_flip_reversal():
    """A4: reversal因子权重取反"""
    w = dict(BASELINE_WEIGHTS)
    for f in REVERSAL_FACTORS:
        if f in w:
            w[f] = -w[f]
    return normalize_weights(w)


def gen_a5_health_weighted():
    """A5: 权重乘以健康度评分"""
    w = dict(BASELINE_WEIGHTS)
    for f in w:
        if f in HEALTH_SCORES:
            w[f] = w[f] * HEALTH_SCORES[f]
    return normalize_weights(w)


EXPERIMENTS = {
    "A0": ("基线(当前实盘权重)", gen_a0_baseline),
    "A1": ("降dead(gtja142/gtja141=0)", gen_a1_zero_dead),
    "A2": ("剔除reversal(8因子=0)", gen_a2_zero_reversal),
    "A3": ("合并A1+A2(10因子=0)", gen_a3_zero_all_alert),
    "A4": ("翻转reversal(8因子取反)", gen_a4_flip_reversal),
    "A5": ("健康度加权(权重×健康度)", gen_a5_health_weighted),
}


def save_weights(name, weights):
    """保存权重到JSON文件"""
    path = os.path.join(WEIGHTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(weights, f, indent=2)
    return path


def run_evaluate(weights_path, start="2026-04-01", end="2026-06-11"):
    """运行评估脚本并返回指标"""
    import subprocess
    result = subprocess.run(
        ["python3", os.path.join(SCRIPT_DIR, "evaluate.py"),
         "--weights", weights_path,
         "--start", start, "--end", end],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  [ERROR] 评估失败: {result.stderr[-200:]}", file=sys.stderr)
        return None
    # 解析stdout最后一行(Calmar)
    lines = result.stdout.strip().split("\n")
    try:
        calmar = float(lines[-1])
    except (ValueError, IndexError):
        print(f"  [ERROR] 无法解析Calmar: {lines}", file=sys.stderr)
        return None
    # 解析stderr中的详细指标
    metrics = {"calmar": calmar}
    for line in result.stderr.split("\n"):
        if "[RESULT]" in line:
            parts = line.split("[RESULT]")[1].strip()
            for p in parts.split():
                if "=" in p:
                    k, v = p.split("=")
                    try:
                        metrics[k] = float(v.rstrip("%"))
                    except ValueError:
                        pass
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="因子衰减权重消融实验")
    parser.add_argument("experiments", nargs="*", default=list(EXPERIMENTS.keys()),
                        help="要运行的实验(默认全部)")
    parser.add_argument("--all-details", action="store_true", help="输出详细指标")
    parser.add_argument("--start", default="2026-04-01", help="回测起始日期")
    parser.add_argument("--end", default="2026-06-11", help="回测结束日期")
    args = parser.parse_args()

    # 生成所有权重文件
    print("=" * 70)
    print("因子衰减权重消融实验")
    print("=" * 70)

    results = {}
    for name in args.experiments:
        if name not in EXPERIMENTS:
            print(f"[WARN] 未知实验: {name}, 跳过")
            continue
        desc, gen_fn = EXPERIMENTS[name]
        weights = gen_fn()
        path = save_weights(name, weights)

        # 统计权重变化
        n_zero = sum(1 for v in weights.values() if abs(v) < 1e-6)
        n_pos = sum(1 for v in weights.values() if v > 1e-6)
        n_neg = sum(1 for v in weights.values() if v < -1e-6)
        print(f"\n{name}: {desc}")
        print(f"  权重: {n_pos}正 {n_neg}负 {n_zero}零, "
              f"abs_sum={sum(abs(v) for v in weights.values()):.4f}")

        # 运行评估
        t0 = time.time()
        metrics = run_evaluate(path, args.start, args.end)
        elapsed = time.time() - t0

        if metrics is None:
            print(f"  [FAIL] 评估失败 ({elapsed:.1f}s)")
            results[name] = {"status": "FAIL", "desc": desc}
            continue

        results[name] = {
            "status": "OK",
            "desc": desc,
            "calmar": metrics.get("calmar", 0),
            "sharpe": metrics.get("SR", 0),
            "mdd": metrics.get("MDD", 0),
            "ar": metrics.get("AR", 0),
            "elapsed": elapsed,
        }
        print(f"  Calmar={metrics.get('calmar', 0):.4f} "
              f"Sharpe={metrics.get('SR', 0):.3f} "
              f"MDD={metrics.get('MDD', 0):.2f}% "
              f"AR={metrics.get('AR', 0):.2f}% ({elapsed:.1f}s)")

    # 汇总对比
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"{'实验':<5} {'描述':<35} {'Calmar':>8} {'Sharpe':>8} {'MDD%':>8} {'AR%':>8} {'状态':>6}")
    print("-" * 80)

    baseline_calmar = results.get("A0", {}).get("calmar", 0)
    for name, r in results.items():
        if r["status"] == "FAIL":
            print(f"{name:<5} {r['desc']:<35} {'---':>8} {'---':>8} {'---':>8} {'---':>8} {'FAIL':>6}")
        else:
            delta = ((r["calmar"] - baseline_calmar) / abs(baseline_calmar) * 100
                     if baseline_calmar != 0 else 0)
            print(f"{name:<5} {r['desc']:<35} {r['calmar']:>8.4f} {r.get('sharpe',0):>8.3f} "
                  f"{r.get('mdd',0):>8.2f} {r.get('ar',0):>8.2f} "
                  f"{'↑' if delta > 0 else '↓' if delta < 0 else '='}")

    # 写入results.tsv
    tsv_path = os.path.join(SCRIPT_DIR, "results.tsv")
    with open(tsv_path, "a") as f:
        for name, r in results.items():
            if r["status"] == "OK":
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts}\t{name}: {r['desc']}\t{r['calmar']:.4f}\t"
                        f"{r.get('sharpe',0):.3f}\t{r.get('mdd',0):.2f}\t"
                        f"{r.get('ar',0):.2f}\tkept\n")
    print(f"\n结果已追加到 {tsv_path}")


if __name__ == "__main__":
    main()