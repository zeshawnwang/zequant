"""批量对比权重梯度方案：V7 → 75% → 50% → 25% → Dead-only(0%)

对比5个权重梯度下Dead因子gtja142/gtja141的表现，找到最优降权幅度。
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_data, evaluate, load_weights, V7_CONFIG

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "weights")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# 5个梯度方案
SCHEMES = [
    ("V7_BASELINE", V7_CONFIG),                              # 100% 原始权重
    ("GTJA142_75PCT", os.path.join(WEIGHT_DIR, "gtja142_75pct.json")),  # 75%
    ("GTJA142_50PCT", os.path.join(WEIGHT_DIR, "gtja142_50pct.json")),  # 50%
    ("GTJA142_25PCT", os.path.join(WEIGHT_DIR, "gtja142_25pct.json")),  # 25%
    ("DEAD_ONLY",    os.path.join(WEIGHT_DIR, "dead_only.json")),       # 0%
]

def main():
    print("=" * 100)
    print("Dead因子权重梯度对比：gtja142 / gtja141")
    print("=" * 100)
    
    # 先加载数据(只加载一次)
    print("\n[1/2] 加载数据...")
    t0 = time.time()
    _ = load_data()
    print(f"  数据加载完成: {time.time()-t0:.1f}s")
    
    print("\n[2/2] 评估5个梯度方案...\n")
    
    results = []
    for label, wpath in SCHEMES:
        print(f"  评估 {label}...", end=" ", flush=True)
        t1 = time.time()
        weights = load_weights(wpath)
        r = evaluate(weights, label)
        elapsed = time.time() - t1
        results.append(r)
        print(f"完成 ({elapsed:.1f}s)")
    
    # 输出对比表
    print("\n" + "=" * 100)
    print("对比结果汇总")
    print("=" * 100)
    
    header = f"{'方案':<18} {'综合分':>8} {'全区间Calmar':>12} {'WF_min':>8} {'2022-24':>8} {'WF非负':>6} {'约束':>4} {'WF1(2022)':>10} {'WF2(2023)':>10} {'WF3(2024)':>10} {'WF4(2025)':>10}"
    print(header)
    print("-" * 100)
    
    for r in results:
        wf = r["wf_oos_calmars"]
        wf_strs = [f"{c:>10.4f}" for c in wf]
        constraint = "✓" if r["constraints_met"] else "✗"
        line = f"{r['label']:<18} {r['score']:>8.4f} {r['full_calmar']:>12.4f} {r['wf_min']:>8.4f} {r['c2022_2024_calmar']:>8.4f} {r['wf_nonneg']:>6d} {constraint:>4}"
        for ws in wf_strs:
            line += f" {ws}"
        print(line)
    
    # 关键指标变化趋势
    print("\n" + "=" * 100)
    print("关键指标趋势 (权重从100%→0%)")
    print("=" * 100)
    
    labels = [r["label"] for r in results]
    calmars = [r["full_calmar"] for r in results]
    wf_mins = [r["wf_min"] for r in results]
    c2022 = [r["c2022_2024_calmar"] for r in results]
    wf1s = [r["wf_oos_calmars"][0] if r["wf_oos_calmars"] else 0 for r in results]
    
    print(f"\n  全区间Calmar:  {' → '.join(f'{c:.4f}' for c in calmars)}")
    print(f"  WF_min:        {' → '.join(f'{c:.4f}' for c in wf_mins)}")
    print(f"  2022-24 Calmar:{' → '.join(f'{c:.4f}' for c in c2022)}")
    print(f"  WF1(2022):     {' → '.join(f'{c:.4f}' for c in wf1s)}")
    
    # 找最优梯度
    print("\n" + "=" * 100)
    print("分析结论")
    print("=" * 100)
    
    # 全区间Calmar最优
    best_full = max(results, key=lambda r: r["full_calmar"])
    # WF_min最优
    best_wf = max(results, key=lambda r: r["wf_min"])
    # 2022-2024最优
    best_2022 = max(results, key=lambda r: r["c2022_2024_calmar"])
    # 综合分最优
    best_score = max(results, key=lambda r: r["score"])
    
    print(f"\n  全区间Calmar最优: {best_full['label']} ({best_full['full_calmar']:.4f})")
    print(f"  WF_min最优:       {best_wf['label']} ({best_wf['wf_min']:.4f})")
    print(f"  2022-24最优:      {best_2022['label']} ({best_2022['c2022_2024_calmar']:.4f})")
    print(f"  综合分最优:       {best_score['label']} ({best_score['score']:.4f})")
    
    # 保存结果
    out_path = os.path.join(RESULTS_DIR, "gradient_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"schemes": results, "best_score_label": best_score["label"]}, f, indent=2, default=str)
    print(f"\n详细结果已保存: {out_path}")

if __name__ == "__main__":
    main()