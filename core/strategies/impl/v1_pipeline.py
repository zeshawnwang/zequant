"""
V1 组件化管道 — V1 核心逻辑的组件编排演示。

展示 V1 的 GA+向量化回测+风险平价如何拆解为项目"器"体系。
每个步骤对应一个明确的组件，通过 import 连接。

管道链路:
  Database → DataMatrixBuilder (3D矩阵) → GAWeightOptimizer (GA搜索)
  → VectorizedEvaluator (回测) → RPPortfolioWeights (仓位分配)
  → RiskConstraints (风控) → 报告输出
"""
import os, sys, json, logging
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core.database import Database
from core.optimization import RiskConstraints, VectorizedEvaluator, EvalResult
from core.positioners import RPPortfolioWeights

logger = logging.getLogger("v1_decoupled")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

TX_COST_RATE = 0.0012
FACTOR_NAMES = ['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20']


# ── 器① 数据源 ──
class DataSource:
    """封装因子和行情数据的加载。"""
    def load(self, start="2018-01-01", end="2026-04-30"):
        logger.info(f"[数据源] 加载因子: {len(FACTOR_NAMES)}个")
        db = Database()
        factor_df = db.get_factors(factor_names=FACTOR_NAMES, start_date=start, end_date=end, with_close=True)
        symbols = db.get_symbols()['symbol'].tolist()
        return factor_df, symbols


# ── 器② 矩阵构建器 ──
class DataMatrixBuilder:
    """将DataFrame转为numpy 3D矩阵 + 截面Z-Score + 前向收益。"""
    def build(self, factor_df, factor_cols, tickers):
        import pandas as pd
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        all_dates = sorted(factor_df['date'].unique())
        n_dates, n_sym, n_factors = len(all_dates), len(tickers), len(factor_cols)
        t2i = {t: i for i, t in enumerate(tickers)}
        d2i = {d: i for i, d in enumerate(all_dates)}

        vals = np.full((n_dates, n_sym, n_factors), np.nan, dtype=np.float32)
        dm = np.zeros((n_dates, n_sym), dtype=bool)
        cl = np.zeros((n_dates, n_sym), dtype=np.float32)

        di = np.array([d2i[d] for d in factor_df['date']], dtype=np.int32)
        si = np.array([t2i.get(s, -1) for s in factor_df['symbol']], dtype=np.int32)
        v = si >= 0; di, si = di[v], si[v]

        for fi, fc in enumerate(factor_cols):
            if fc in factor_df.columns:
                vals[di, si, fi] = factor_df[fc].values[v].astype(np.float32)
        cl[di, si] = factor_df['close'].values[v].astype(np.float32)
        dm[di, si] = True
        np.nan_to_num(vals, nan=0.0, copy=False)
        np.nan_to_num(cl, nan=0.0, copy=False)

        fwd = np.zeros((n_dates, n_sym), dtype=np.float32)
        for d in range(n_dates - 1):
            b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
            fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

        z3d = np.zeros_like(vals)
        for fi in range(n_factors):
            a = vals[:, :, fi]
            for d in range(n_dates):
                r = a[d, :]
                nz = r[r != 0]
                lo, hi = np.quantile(nz, [0.01, 0.99]) if len(nz) > 0 else (0, 0)
                c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z3d[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0

        logger.info(f"[矩阵构建] {n_dates}天 × {n_sym}只 × {n_factors}因子")
        return z3d, fwd, dm, all_dates


# ── 器③ GA权重优化器 ──
class GAWeightOptimizer:
    """遗传算法权重搜索（V1核心）。"""
    def __init__(self, ngen=50, npop=30, l1_lambda=0.003, rebal_freq=3,
                 mutate_prob=0.15, crossover_prob=0.7, stall_limit=15, top_n=30):
        self.ngen = ngen; self.npop = npop; self.l1_lambda = l1_lambda
        self.rebal_freq = rebal_freq; self.mutate_prob = mutate_prob
        self.crossover_prob = crossover_prob; self.stall_limit = stall_limit
        self.top_n = top_n

    def optimize(self, z_3d, fwd_rets, data_mask, risk_ctrl=None, n_results=5):
        np.random.seed(42)
        nf = z_3d.shape[2]
        ev = VectorizedEvaluator(tx_cost_rate=TX_COST_RATE,
            portfolio_builder=RPPortfolioWeights(top_n=self.top_n, min_hold_days=5))
        n_total = z_3d.shape[0]; te = int(n_total * 0.75)
        tz, tf = z_3d[:te], fwd_rets[:te]; tm = data_mask[:te] if data_mask is not None else None

        pop = np.random.uniform(-0.5, 0.5, (self.npop, nf)).astype(np.float32)
        fit = np.full(self.npop, -np.inf, dtype=np.float32); bf = -np.inf; bw = None; st = 0
        for gen in range(self.ngen):
            for i in range(self.npop):
                if not np.isfinite(fit[i]):
                    s = ev.evaluate_with_risk_check(pop[i], tz, tf, tm, self.rebal_freq, self.top_n, self.l1_lambda, 0, risk_ctrl)
                    fit[i] = s if np.isfinite(s) else 0.0
            cb = float(np.max(fit)); bi = int(np.argmax(fit))
            if cb > bf: bf = cb; bw = pop[bi].copy(); st = 0
            else: st += 1
            logger.info(f"  [GA] 第{gen+1:3d}/{self.ngen}代 best={cb:.4f} ever={bf:.4f} stall={st}")
            if st >= self.stall_limit: break
            fit[:] = -np.inf
            if gen == self.ngen - 1: break
            np_, ef_ = [], {}
            for idx in np.argsort(-fit)[:max(2, self.npop // 10)]:
                if np.isfinite(fit[idx]): np_.append(pop[idx].copy()); ef_[len(np_)-1] = fit[idx]
            while len(np_) < self.npop:
                if np.random.random() < self.crossover_prob:
                    i1, i2 = np.random.randint(0, self.npop, 2)
                    a = np.random.random(nf).astype(np.float32)
                    np_.append(a * pop[i1] + (1 - a) * pop[i2])
                    if len(np_) < self.npop: np_.append(a * pop[i2] + (1 - a) * pop[i1])
                else: np_.append(pop[np.random.randint(0, self.npop)].copy())
            pop = np.array(np_[:self.npop], dtype=np.float32)
            mm = np.random.random(pop.shape) < self.mutate_prob
            pop[mm] += np.random.normal(0, 0.1, pop.shape)[mm]
            pop = np.clip(pop, -0.5, 0.5)
            nf2 = np.full(self.npop, -np.inf, dtype=np.float32)
            for k, v in ef_.items(): nf2[k] = v
            fit = nf2

        scores = [(ev.evaluate_with_risk_check(pop[i], tz, tf, tm, self.rebal_freq, self.top_n, self.l1_lambda, 0, risk_ctrl), pop[i].copy()) for i in range(self.npop)]
        scores.sort(key=lambda x: x[0], reverse=True)
        res = [{"score": float(s), "weights": w.tolist()} for s, w in scores[:n_results] if s > -1]
        if bw is not None:
            s = ev.evaluate_with_risk_check(bw, tz, tf, tm, self.rebal_freq, self.top_n, self.l1_lambda, 0, risk_ctrl)
            if s > -1: res.insert(0, {"score": float(s), "weights": bw.tolist()})
        logger.info(f"[GA优化] 完成! {len(res)}组有效结果")
        return res


# ── 器④ 回测验证器 ──
class BacktestValidator:
    """全样本回测验证。"""
    def __init__(self):
        self.ev = VectorizedEvaluator(tx_cost_rate=TX_COST_RATE,
            portfolio_builder=RPPortfolioWeights(top_n=40, min_hold_days=5))

    def run(self, weights, z_3d, fwd_rets, data_mask, name="") -> dict:
        r = self.ev.evaluate(np.array(weights, dtype=np.float32), z_3d, fwd_rets, data_mask, rebal_freq=3, top_n=40, return_equity=True)
        logger.info(f"[回测] {name}: 年化={r.annual_return*100:.2f}% Sharpe={r.sharpe:.3f} 回撤={r.max_drawdown*100:.2f}%")
        return {"name": name, "annual_return": r.annual_return, "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown, "calmar": r.calmar, "win_rate": r.win_rate}


# ── 管道入口 ──
def run_v1_pipeline():
    logger.info("=" * 60)
    logger.info("V1 Decoupled Pipeline")
    logger.info("链路: DataSource → MatrixBuilder → GAOptimizer → BacktestValidator")
    logger.info(f"组件: Database | RPPortfolioWeights | VectorizedEvaluator | RiskConstraints")
    logger.info("=" * 60)

    ds = DataSource()
    factor_df, tickers = ds.load()

    meta_cols = {'date','symbol','open','high','low','close','volume','amount','pct_change'}
    factor_cols = [c for c in FACTOR_NAMES if c in factor_df.columns]

    mb = DataMatrixBuilder()
    z3d, fwd, dm, dates = mb.build(factor_df, factor_cols, tickers)

    rc = RiskConstraints(max_drawdown=0.90, min_calmar_ratio=0.0, min_win_rate=0.0)
    ga = GAWeightOptimizer(ngen=3, npop=5, l1_lambda=0.003, rebal_freq=3, stall_limit=10)

    results = ga.optimize(z3d, fwd, dm, rc, n_results=3)
    if not results:
        logger.warning("GA无有效结果"); return

    bv = BacktestValidator()
    bt_results = [bv.run(r['weights'], z3d, fwd, dm, f"配置_{r['score']:.4f}") for r in results]

    logger.info("=" * 60)
    logger.info("管道完成! 结果:")
    for bt in sorted(bt_results, key=lambda x: x['sharpe'], reverse=True):
        logger.info(f"  {bt['name']}: 年化={bt['annual_return']*100:.2f}% Sharpe={bt['sharpe']:.3f} 回撤={bt['max_drawdown']*100:.2f}%")

    summary = {"version": "v1_decoupled", "parameters": {"factors": len(factor_cols), "tx_cost": TX_COST_RATE},
               "results": bt_results, "date_range": f"{dates[0]}~{dates[-1]}"}
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "daily", "2026-05-13", "v1", "decoupled_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"结果: {out_path}")


if __name__ == "__main__":
    run_v1_pipeline()
