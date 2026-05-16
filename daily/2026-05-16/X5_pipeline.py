"""
X5_pipeline — 全量GA二次优化

使用VectorizedEvaluator + GA对 ~70个高价值因子做权重优化:
  - 50个V1最佳因子 + 16个新技术因子 (ma5/ma20/ma_alignment_score 等)
  - 简化GA: npop=30, ngen=50, l1_lambda=0.005
  - 评估: VectorizedEvaluator.evaluate_with_risk_check()

用法:
    python3 daily/2026-05-16/X5_pipeline.py 2>&1 | tee daily/2026-05-16/x4_x5_results/x5.log
"""
import os, sys, json, logging, gc, time, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.optimization import VectorizedEvaluator, RiskConstraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("x5")
TX = 0.0012

OUT_DIR = os.path.join(os.path.dirname(__file__), "x4_x5_results")
os.makedirs(OUT_DIR, exist_ok=True)

# V1 50个最佳因子
_V1_BEST = [
    'ff_mkt','gtja142','gtja144','gtja171','gtja103','gtja85','a88','a31',
    'rsi_14','gtja139','gtja123','a42','a41','a97','gtja148','gtja99',
    'gtja117','gtja76','gtja90','volatility_20','gtja113','gtja141','a99',
    'gtja12','gtja83','gtja164','a98','gtja49','gtja121','a85','gtja104',
    'gtja185','gtja176','a80','gtja62','a8','gtja34','returns','gtja168',
    'gtja108','gtja105','gtja127','a27','a64','gtja91','a30','a69','a91',
    'gtja13','gtja120',
]

# 16个新技术因子
_NEW_TECH = [
    'ma5','ma20','ma60','ma120',
    'ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross',
    'ma_angle_20','volume_breakout_ratio','volume_contraction',
    'chip_concentration','ma_convergence',
    'box_breakout','breakout_strength',
]

# 合并因子
ALL_FACTORS = list(dict.fromkeys(_V1_BEST + _NEW_TECH))
logger.info(f"X5因子总数: {len(ALL_FACTORS)} (V1: {len(_V1_BEST)} + 新技术: {len(_NEW_TECH)})")


# ============================================================
# 数据加载
# ============================================================
def load():
    """加载因子数据 + fwd收益"""
    db = Database()

    df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                        factor_names=ALL_FACTORS, with_close=True)
    df['date'] = pd.to_datetime(df['date'])
    ds = sorted(df['date'].unique())
    tks = db.get_symbols()['symbol'].tolist()
    nd, ns, nf = len(ds), len(tks), len(ALL_FACTORS)
    t2i = {t: i for i, t in enumerate(tks)}
    d2i = {d: i for i, d in enumerate(ds)}

    v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32)
    dm = np.zeros((nd, ns), dtype=bool)
    cl = np.zeros((nd, ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
    for fi, fc in enumerate(ALL_FACTORS):
        if fc in df.columns:
            v3[di, si, fi] = df[fc].values[v].astype(np.float32)
    cl[di, si] = df['close'].values[v].astype(np.float32)
    dm[di, si] = True
    np.nan_to_num(v3, nan=0.0, copy=False)
    np.nan_to_num(cl, nan=0.0, copy=False)

    # 前向收益
    fwd = np.zeros((nd, ns), dtype=np.float32)
    for d in range(nd - 1):
        b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
        fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

    # 截面Z-Score
    z3 = np.zeros_like(v3)
    for fi in range(nf):
        a = v3[:, :, fi]
        for d in range(nd):
            r = a[d, :]; nz = r[r != 0]
            if len(nz) > 1:
                lo, hi = np.quantile(nz, [0.01, 0.99]); c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z3[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0

    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
    return z3, fwd, dm, tks, list(ALL_FACTORS), nd, ns, ds, t2i, d2i


# ============================================================
# 简化GA优化
# ============================================================
class SimpleGAOptimizer:
    """简化版遗传算法权重优化器。

    使用VectorizedEvaluator.evaluate_with_risk_check()评估适应度。
    """

    def __init__(self, factor_z, fwd_ret, data_mask, n_factors,
                 npop=30, ngen=50, l1_lambda=0.005, rebal_freq=3,
                 top_n=40, mutation_rate=0.15, crossover_rate=0.7,
                 elitism=3):
        self.factor_z = factor_z
        self.fwd_ret = fwd_ret
        self.data_mask = data_mask
        self.n_factors = n_factors
        self.npop = npop
        self.ngen = ngen
        self.l1_lambda = l1_lambda
        self.rebal_freq = rebal_freq
        self.top_n = top_n
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        self.evaluator = VectorizedEvaluator(tx_cost_rate=TX)
        self.risk_constraints = RiskConstraints(
            max_drawdown=0.20,
            min_calmar_ratio=0.5,
            min_win_rate=0.48,
        )

        self.best_history = []
        self.population = []
        self.fitness = []

    def _init_population(self):
        """初始化种群: 正态分布权重 + 随机稀疏化"""
        pop = []
        for _ in range(self.npop):
            w = np.random.normal(0, 0.5, self.n_factors)
            # 随机将部分权重置0 (稀疏化)
            sparse_mask = np.random.random(self.n_factors) < 0.4
            w[sparse_mask] = 0.0
            if np.all(w == 0):
                w[np.random.randint(self.n_factors)] = 1.0
            pop.append(w)
        return pop

    def _evaluate(self, weights):
        return self.evaluator.evaluate_with_risk_check(
            weights, self.factor_z, self.fwd_ret, self.data_mask,
            rebal_freq=self.rebal_freq, top_n=self.top_n,
            l1_lambda=self.l1_lambda, risk_constraints=self.risk_constraints,
        )

    def _tournament_select(self, fitness, k=3):
        """锦标赛选择"""
        idx = random.sample(range(len(fitness)), k)
        best = max(idx, key=lambda i: fitness[i])
        return best

    def _crossover(self, p1, p2):
        """模拟二进制交叉 (SBX)"""
        child = np.zeros_like(p1)
        for i in range(self.n_factors):
            if random.random() < 0.5:
                child[i] = p1[i]
            else:
                child[i] = p2[i]
        return child

    def _mutate(self, w):
        """高斯变异"""
        for i in range(self.n_factors):
            if random.random() < self.mutation_rate:
                w[i] += np.random.normal(0, 0.2)
        return w

    def run(self):
        t0 = time.time()
        logger.info(f"GA开始: {self.n_factors}因子 × {self.npop}种群 × {self.ngen}代")

        self.population = self._init_population()

        for gen in range(self.ngen):
            # 评估
            self.fitness = [self._evaluate(w) for w in self.population]
            valid_f = [f for f in self.fitness if f > -0.5]

            if valid_f:
                avg_f = np.mean(valid_f)
                best_f = max(valid_f)
            else:
                avg_f = -1.0
                best_f = -1.0
            self.best_history.append(best_f)

            if (gen + 1) % 10 == 0 or gen == 0:
                elapsed = time.time() - t0
                logger.info(f"  代 {gen+1}/{self.ngen} | best={best_f:.4f} avg={avg_f:.4f} | "
                           f"有效={len(valid_f)}/{self.npop} | {elapsed:.0f}s")

            # 精英保留
            sorted_idx = np.argsort(self.fitness)[::-1]
            next_pop = [self.population[i].copy() for i in sorted_idx[:self.elitism]]

            # 生成新个体
            while len(next_pop) < self.npop:
                i1 = self._tournament_select(self.fitness)
                i2 = self._tournament_select(self.fitness)
                p1, p2 = self.population[i1], self.population[i2]

                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.copy()

                child = self._mutate(child)
                next_pop.append(child)

            self.population = next_pop

            gc.collect()

        # 最终评估
        final_fitness = [self._evaluate(w) for w in self.population]
        best_idx = int(np.argmax(final_fitness))
        best_weights = self.population[best_idx]
        best_score = final_fitness[best_idx]

        elapsed = time.time() - t0
        logger.info(f"GA完成! 耗时 {elapsed/60:.1f}分 | 最佳得分={best_score:.4f}")
        return best_weights, best_score, self.best_history


def run_ga_full(z3, fwd, dm, factor_names):
    """运行完整GA优化"""
    nd, ns, nf = z3.shape
    logger.info(f"\n{'='*70}")
    logger.info(f"任务2: 全量GA二次优化 ({nf}个因子)")
    logger.info(f"{'='*70}")

    # 训练期: 2019-01 ~ 2024-06
    ds = np.array([pd.Timestamp(d) for d in range(nd)])
    # 用日期索引
    from datetime import datetime
    # 直接用数据切片, 前20%做warmup, 后面做训练
    train_start = int(nd * 0.2)  # ~2020年
    train_end = int(nd * 0.75)   # ~2024年中

    z3_train = z3[train_start:train_end]
    fwd_train = fwd[train_start:train_end]
    dm_train = dm[train_start:train_end]

    # 跑3种rebal_freq配置
    best_results = []
    for rf in [3, 5, 10]:
        logger.info(f"\n--- GA Rebal Freq={rf} ---")
        opt = SimpleGAOptimizer(
            z3_train, fwd_train, dm_train, nf,
            npop=30, ngen=50, l1_lambda=0.005,
            rebal_freq=rf, top_n=40,
        )
        best_w, best_score, history = opt.run()

        # 在验证期回测 (2024-07 ~ 2026-04)
        val_start = train_end
        val_z3, val_fwd, val_dm = z3[val_start:], fwd[val_start:], dm[val_start:]
        evaluator = VectorizedEvaluator(tx_cost_rate=TX)
        val_result = evaluator.evaluate(best_w, val_z3, val_fwd, val_dm,
                                         rebal_freq=rf, top_n=40, return_equity=True)

        logger.info(f"  rf={rf} | 训练得分={best_score:.4f} | "
                    f"验证年化={val_result.annual_return*100:.2f}% "
                    f"Sharpe={val_result.sharpe:.3f} "
                    f"回撤={val_result.max_drawdown*100:.2f}%")

        # 非零权重
        nz_mask = np.abs(best_w) > 1e-6
        nz_count = int(np.sum(nz_mask))
        nz_factors = [factor_names[i] for i in range(nf) if nz_mask[i]]
        nz_weights = {factor_names[i]: float(best_w[i]) for i in range(nf) if nz_mask[i]}

        best_results.append({
            'rebal_freq': rf,
            'best_score': float(best_score),
            'n_nonzero': nz_count,
            'nonzero_factors': nz_factors,
            'weights': nz_weights,
            'train': {
                'annual_return': None,  # 未单独保存训练结果, 可后续run
                'sharpe': None,
                'max_drawdown': None,
            },
            'validation': {
                'annual_return': val_result.annual_return,
                'sharpe': val_result.sharpe,
                'max_drawdown': val_result.max_drawdown,
                'calmar': val_result.calmar,
                'win_rate': val_result.win_rate,
                'n_trades': val_result.n_trades,
            },
            'history': history,
        })

        gc.collect()

    return best_results


# ============================================================
# 对比: V1权重在新因子集上的表现
# ============================================================
def run_v1_baseline(z3_train, fwd_train, dm_train,
                    z3_val, fwd_val, dm_val, factor_names):
    """用V1 L1中权重在新因子集上做对比"""
    logger.info("\n--- V1 Baseline (仅含V1因子) ---")

    # 加载V1权重
    v1w_path = os.path.join(os.path.dirname(__file__), '..', '2026-05-13',
                            'v2', 'v1_reference', 'ga_results.json')
    if not os.path.exists(v1w_path):
        logger.warning("V1权重文件不存在,跳过baseline")
        return []

    with open(v1w_path) as f:
        for item in json.load(f):
            if 'L1中_80代' in item['label']:
                v1_weights = item['configs'][0]['weights']
                break
        else:
            return []

    # 构建V1权重向量 (只对V1部分, 新因子权重为0)
    w_v1 = np.zeros(len(factor_names), dtype=np.float32)
    for fi, fn in enumerate(factor_names):
        if fn in v1_weights:
            w_v1[fi] = float(v1_weights[fn])
    s = np.sum(np.abs(w_v1))
    if s > 0:
        w_v1 /= s

    evaluator = VectorizedEvaluator(tx_cost_rate=TX)
    results = []
    for rf in [3, 5, 10]:
        # 训练期
        tr = evaluator.evaluate(w_v1, z3_train, fwd_train, dm_train,
                                 rebal_freq=rf, top_n=40)
        # 验证期
        vr = evaluator.evaluate(w_v1, z3_val, fwd_val, dm_val,
                                 rebal_freq=rf, top_n=40)
        results.append({
            'config': f'V1_Baseline_rf{rf}',
            'train_ar': tr.annual_return,
            'train_sharpe': tr.sharpe,
            'train_mdd': tr.max_drawdown,
            'val_ar': vr.annual_return,
            'val_sharpe': vr.sharpe,
            'val_mdd': vr.max_drawdown,
        })
        logger.info(f"  V1_rf{rf} | 训练: {tr.annual_return*100:.2f}%/{tr.sharpe:.3f} | "
                    f"验证: {vr.annual_return*100:.2f}%/{vr.sharpe:.3f}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("X5 Pipeline — 全量GA二次优化")
    logger.info("=" * 70)

    t_start = time.time()

    # 1. 加载数据
    z3, fwd, dm, tks, fnames, nd, ns, ds, t2i, d2i = load()
    logger.info(f"数据加载完成: {nd}天 × {ns}只 × {len(fnames)}因子")

    # 2. 分割训练/验证
    train_end = int(nd * 0.75)  # 2018 ~ 2024年中的75%
    z3_train, fwd_train, dm_train = z3[:train_end], fwd[:train_end], dm[:train_end]
    z3_val, fwd_val, dm_val = z3[train_end:], fwd[train_end:], dm[train_end:]
    logger.info(f"训练: {train_end}天, 验证: {nd - train_end}天")

    # 3. V1 Baseline
    baseline_results = run_v1_baseline(z3_train, fwd_train, dm_train,
                                        z3_val, fwd_val, dm_val, fnames)

    # 4. GA优化
    ga_results = run_ga_full(z3, fwd, dm, fnames)

    # 5. 输出GA优化结果
    best_weights_export = []
    for r in ga_results:
        best_weights_export.append({
            'rebal_freq': r['rebal_freq'],
            'best_score': r['best_score'],
            'n_nonzero': r['n_nonzero'],
            'nonzero_factors': r['nonzero_factors'],
            'weights': r['weights'],
            'validation': r['validation'],
        })

    # 6. 保存结果
    out_path = os.path.join(OUT_DIR, "x5_results.json")
    output = {
        'factor_count': len(fnames),
        'v1_best_factors': _V1_BEST,
        'new_tech_factors': _NEW_TECH,
        'baseline': baseline_results,
        'ga_results': best_weights_export,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存至: {out_path}")

    # 7. 打印汇总
    print(f"\n{'='*110}")
    print(f"{'配置':<30} {'训练Sharpe':<12} {'训练年化%':<10} {'验证Sharpe':<12} {'验证年化%':<10} {'验证回撤%':<10}")
    print('-'*110)
    for r in baseline_results:
        print(f"  {r['config']:<28} {r['train_sharpe']:>8.3f}   {r['train_ar']*100:>7.2f}% "
              f"{r['val_sharpe']:>8.3f}   {r['val_ar']*100:>7.2f}%   {r['val_mdd']*100:>6.2f}%")
    print('-'*110)
    for r in best_weights_export:
        print(f"  GA_rf{r['rebal_freq']:<24} {'-':>8}   {'-':>7}  "
              f"{r['validation']['sharpe']:>8.3f}   {r['validation']['annual_return']*100:>7.2f}%   "
              f"{r['validation']['max_drawdown']*100:>6.2f}%")
    print('='*110)

    # 打印GA重要因子
    print(f"\n{'='*70}")
    print("GA优化输出权重 (非零因子)")
    for r in best_weights_export:
        print(f"\n--- RF={r['rebal_freq']} | 非零={r['n_nonzero']} | 得分={r['best_score']:.4f} ---")
        sorted_w = sorted(r['weights'].items(), key=lambda x: abs(x[1]), reverse=True)
        for fn, fw in sorted_w[:20]:
            print(f"    {fn:<20} = {fw:>+.6f}")
    print('='*70)

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nX5 全部完成! 总耗时 {elapsed:.1f}分")


if __name__ == "__main__":
    main()
