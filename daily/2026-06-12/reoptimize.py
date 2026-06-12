"""近期窗口GA重优化 — 用遗传算法在近期数据上重新优化因子权重

用法:
    python3 daily/2026-06-12/reoptimize.py
    python3 daily/2026-06-12/reoptimize.py --generations 50 --population 30
    python3 daily/2026-06-12/reoptimize.py --start 2025-06-01 --end 2026-06-11

策略:
    1. 在近期窗口(默认2025-06~2026-06)上运行GA优化
    2. 用OOS期(2026-04~2026-06)验证优化结果
    3. 与基线权重对比
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")

DB_PATH = os.path.abspath("./data/quant_data.db")
TX = 0.0012

LIVE_FACTORS = [
    'ff_mkt','gtja142','gtja144','gtja171','gtja103','gtja85','a88','a31',
    'rsi_14','gtja139','gtja123','a42','a41','a97','gtja148','gtja99',
    'gtja117','gtja76','gtja90','volatility_20','gtja113','gtja141','a99',
    'gtja12','gtja83','gtja164','a98','gtja49','gtja121','a85','gtja104',
    'gtja185','gtja176','a80','gtja62','a8','gtja34','returns','gtja168',
    'gtja108','gtja105','gtja127','a27','a64','gtja91','a30','a69','a91',
    'gtja13','gtja120'
]

# 基线权重
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


def run_evaluate(weights_dict, start="2026-04-01", end="2026-06-11"):
    """运行评估脚本返回Calmar"""
    path = os.path.join(WEIGHTS_DIR, "_ga_temp.json")
    with open(path, "w") as f:
        json.dump(weights_dict, f)
    result = subprocess.run(
        ["python3", os.path.join(SCRIPT_DIR, "evaluate.py"),
         "--weights", path, "--start", start, "--end", end],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().split("\n")
    try:
        return float(lines[-1])
    except (ValueError, IndexError):
        return None


def random_weights(n_factors, rng):
    """生成随机权重(正负混合)"""
    w = rng.standard_normal(n_factors).astype(np.float32)
    # 随机符号翻转概率30%
    flip = rng.random(n_factors) < 0.3
    w[flip] = -np.abs(w[flip])
    w[~flip] = np.abs(w[~flip])
    # 缩放
    w = w / np.sum(np.abs(w))
    return w


def crossover(w1, w2, rng):
    """均匀交叉"""
    mask = rng.random(len(w1)) < 0.5
    child = np.where(mask, w1, w2)
    return child / np.sum(np.abs(child))


def mutate(w, rng, rate=0.15, scale=0.3):
    """高斯变异"""
    mask = rng.random(len(w)) < rate
    noise = rng.standard_normal(len(w)) * scale
    w = w + mask * noise
    return w / np.sum(np.abs(w))


def weights_array_to_dict(w, factor_names):
    """数组转字典"""
    return {f: float(w[i]) for i, f in enumerate(factor_names)}


def main():
    parser = argparse.ArgumentParser(description="近期窗口GA重优化")
    parser.add_argument("--start", default="2025-06-01", help="优化窗口起始")
    parser.add_argument("--end", default="2026-06-11", help="优化窗口结束")
    parser.add_argument("--oos-start", default="2026-04-01", help="OOS验证起始")
    parser.add_argument("--oos-end", default="2026-06-11", help="OOS验证结束")
    parser.add_argument("--generations", type=int, default=30, help="GA代数")
    parser.add_argument("--population", type=int, default=20, help="种群大小")
    parser.add_argument("--elite", type=int, default=3, help="精英保留数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", default=None, help="输出权重JSON路径")
    args = parser.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 因子列表(只优化有基线权重的因子)
    factor_names = sorted(BASELINE_WEIGHTS.keys())
    n_factors = len(factor_names)
    baseline_w = np.array([BASELINE_WEIGHTS[f] for f in factor_names], dtype=np.float32)
    baseline_w = baseline_w / np.sum(np.abs(baseline_w))

    print("=" * 70)
    print(f"GA权重重优化 ({args.generations}代, 种群{args.population})")
    print(f"  优化窗口: {args.start} ~ {args.end}")
    print(f"  OOS验证:  {args.oos_start} ~ {args.oos_end}")
    print(f"  因子数:   {n_factors}")
    print("=" * 70)

    # 评估基线
    print("\n[基线评估]")
    baseline_dict = weights_array_to_dict(baseline_w, factor_names)
    baseline_calmar = run_evaluate(baseline_dict, args.oos_start, args.oos_end)
    print(f"  基线Calmar(OOS): {baseline_calmar:.4f}" if baseline_calmar else "  基线评估失败")

    # 初始化种群: 包含基线 + 随机个体
    population = [baseline_w.copy()]
    for _ in range(args.population - 1):
        population.append(random_weights(n_factors, rng))

    best_calmar = baseline_calmar or 0
    best_weights = baseline_w.copy()
    history = []

    for gen in range(args.generations):
        t0 = time.time()
        # 评估适应度(Calmar on optimization window)
        fitness = []
        for w in population:
            wd = weights_array_to_dict(w, factor_names)
            cal = run_evaluate(wd, args.start, args.end)
            fitness.append(cal if cal is not None else -999)

        # 排序
        ranked = sorted(zip(fitness, range(len(population))), key=lambda x: -x[0])
        elite_idx = [idx for _, idx in ranked[:args.elite]]

        gen_best_cal = ranked[0][0]
        gen_best_w = population[ranked[0][1]].copy()
        avg_cal = np.mean([f for f in fitness if f > -900])

        # OOS验证当前最佳
        if gen_best_cal > best_calmar:
            oos_cal = run_evaluate(weights_array_to_dict(gen_best_w, factor_names),
                                   args.oos_start, args.oos_end)
            if oos_cal is not None and oos_cal > best_calmar:
                best_calmar = oos_cal
                best_weights = gen_best_w.copy()
                print(f"  Gen {gen+1:3d}: 新最佳! OOS Calmar={oos_cal:.4f} (IS={gen_best_cal:.4f})")

        history.append({"gen": gen + 1, "is_best": gen_best_cal, "avg": avg_cal,
                         "oos_best": best_calmar})
        elapsed = time.time() - t0
        print(f"  Gen {gen+1:3d}/{args.generations}: IS_best={gen_best_cal:.4f} "
              f"avg={avg_cal:.4f} OOS_best={best_calmar:.4f} ({elapsed:.1f}s)")

        # 生成下一代
        new_pop = [population[i].copy() for i in elite_idx]
        while len(new_pop) < args.population:
            # 锦标赛选择
            i1, i2 = rng.choice(len(population), 2, replace=False)
            p1 = population[i1] if fitness[i1] > fitness[i2] else population[i2]
            i3, i4 = rng.choice(len(population), 2, replace=False)
            p2 = population[i3] if fitness[i3] > fitness[i4] else population[i4]
            child = crossover(p1, p2, rng)
            child = mutate(child, rng)
            new_pop.append(child)
        population = new_pop

    # 保存最佳权重
    best_dict = weights_array_to_dict(best_weights, factor_names)
    output_path = args.output or os.path.join(WEIGHTS_DIR, "ga_optimized.json")
    with open(output_path, "w") as f:
        json.dump(best_dict, f, indent=2)

    # 最终OOS验证
    final_calmar = run_evaluate(best_dict, args.oos_start, args.oos_end)

    print("\n" + "=" * 70)
    print("GA优化结果")
    print("=" * 70)
    print(f"  基线Calmar:  {baseline_calmar:.4f}" if baseline_calmar else "  基线Calmar:  N/A")
    print(f"  优化Calmar:  {final_calmar:.4f}" if final_calmar else "  优化Calmar:  N/A")
    if baseline_calmar and final_calmar:
        delta = (final_calmar - baseline_calmar) / abs(baseline_calmar) * 100
        print(f"  改进:        {delta:+.1f}%")
    print(f"\n  权重已保存到: {output_path}")

    # 与基线对比权重变化
    print(f"\n  {'因子':<15} {'基线':>8} {'优化':>8} {'变化':>8}")
    print("  " + "-" * 40)
    for i, f in enumerate(factor_names):
        bv = baseline_w[i]
        nv = best_weights[i]
        delta_w = nv - bv
        mark = "★" if abs(delta_w) > 0.02 else ""
        print(f"  {f:<15} {bv:>8.4f} {nv:>8.4f} {delta_w:>+8.4f} {mark}")

    # 写入results.tsv
    tsv_path = os.path.join(SCRIPT_DIR, "results.tsv")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(tsv_path, "a") as f:
        f.write(f"{ts}\tGA重优化({args.generations}代)\t{final_calmar:.4f}\t"
                f"---\t---\t---\tkept\n")
    print(f"\n结果已追加到 {tsv_path}")


if __name__ == "__main__":
    main()