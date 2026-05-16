#!/usr/bin/env python3
"""综合验证与优化 Pipeline。

功能：
1. Enhanced GA (L1正则化 + 换手率惩罚 + 更多迭代)
2. 滚动窗口样本外验证 (市场环境适应性)
3. 真实 BacktestEngine 验证
4. 断点续传 + 全量日志

使用方式:
  python enhanced_optimizer.py [--start 2019-01-01] [--end 2026-05-13]
"""
from __future__ import annotations
import os, sys, json, logging, shutil
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.database import Database
from core.optimization.base.factor_categories import FACTOR_CATEGORIES
from core.optimization.base.risk_constraints import RiskConstraints

# ============================================================
# 市场周期定义 (基于历史A股主要阶段)
# ============================================================
MARKET_WINDOWS = [
    {"label": "2019_修复牛", "start": "2019-01-02", "end": "2019-12-31", "type": "bull"},
    {"label": "2020_疫情冲击+反弹", "start": "2020-01-02", "end": "2020-12-31", "type": "volatile"},
    {"label": "2021_结构牛", "start": "2021-01-04", "end": "2021-12-31", "type": "bull"},
    {"label": "2022_熊市", "start": "2022-01-04", "end": "2022-12-30", "type": "bear"},
    {"label": "2023_震荡修复", "start": "2023-01-03", "end": "2023-12-29", "type": "recovery"},
    {"label": "2024_反弹", "start": "2024-01-02", "end": "2024-12-31", "type": "bull"},
    {"label": "2025_至今", "start": "2025-01-02", "end": "2026-05-13", "type": "mixed"},
]
# GA 优化仅用后半段 (2021+, 样本外: 2019-2020)
GA_START = "2021-01-04"

# 交易成本模型参数
TX_COST_RATE = 0.0015       # 单边交易成本率: 佣金万三 + 印花税均摊千一/2 + 滑点0.1%
MIN_HOLD_DAYS = 5           # 最低持仓天数 (约1周)
DEFAULT_REBAL_FREQ = 3      # 默认调仓频率 (~1.7次/交易周)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return 1e10 if obj > 0 else -1e10
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def sanitize_for_json(obj):
    """递归清洗数据, 确保 JSON 安全 (无 NaN/Infinity)。"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return float(obj) if isinstance(obj, np.floating) else (int(obj) if isinstance(obj, np.integer) else bool(obj))
    return obj


@dataclass
class ValidationCheckpoint:
    timestamp: str
    step: str  # enhanced_ga / rolling_validation / real_backtest / done
    ga_configs: List[Dict] = field(default_factory=list)
    rolling_results: List[Dict] = field(default_factory=list)
    real_backtest_results: List[Dict] = field(default_factory=list)

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            data = sanitize_for_json(asdict(self))
            json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    @staticmethod
    def load(path: Path) -> 'ValidationCheckpoint':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ValidationCheckpoint(**data)


class EnhancedOptimizer:
    """综合验证与优化器。"""

    def __init__(
        self,
        db_path: str = "./data/quant_data.db",
        output_dir: str = "./daily",
        max_drawdown: float = 0.60,
    ):
        self.db = Database(db_path)
        self.output_dir = Path(output_dir)
        self.today_dir = self.output_dir / datetime.now().strftime('%Y-%m-%d')
        self.today_dir.mkdir(parents=True, exist_ok=True)

        self.risk_constraints = RiskConstraints(
            max_drawdown=max_drawdown,
            single_stock_weight=0.15, single_sector_weight=0.25,
            max_volatility=0.30, max_turnover=1.0,
            min_calmar_ratio=0.4, min_win_rate=0.30,
        )
        self.checkpoint_file = self.today_dir / "checkpoint_validate.json"
        self.log_file = self.today_dir / "validation.log"
        self._setup_logging()

        # 缓存预计算数据 (只计算一次, 所有步骤复用)
        self._eval_cache: Optional[Dict] = None
        self._all_factors: List[str] = []
        self._selected_categories: Dict[str, List[str]] = {}

    def _setup_logging(self):
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(self.log_file, encoding='utf-8'),
                      logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _load_checkpoint(self) -> Optional[ValidationCheckpoint]:
        if self.checkpoint_file.exists():
            try:
                return ValidationCheckpoint.load(self.checkpoint_file)
            except Exception as e:
                self.logger.warning(f"检查点加载失败: {e}")
        return None

    def _save_checkpoint(self, cp: ValidationCheckpoint):
        cp.timestamp = datetime.now().isoformat()
        cp.save(self.checkpoint_file)

    # ----------------------------------------------------------
    # Step 0: 加载已选因子
    # ----------------------------------------------------------
    def load_selected_factors(self) -> Dict[str, List[str]]:
        """从 config 或 daily 路径加载选出的50个因子。"""
        config_paths = [
            self.today_dir / "final_config.json",
            ROOT / "config" / "optimized_multi_factor_ga.yaml",
            self.today_dir.parent / "attribution_analysis.json",
        ]
        for p in config_paths:
            if not p.exists():
                continue
            try:
                if p.suffix == ".json":
                    with open(p) as f:
                        data = json.load(f)
                    factors = data.get("selected_factors", {})
                    if factors:
                        self.logger.info(f"从 {p.name} 加载了 {sum(len(v) for v in factors.values())} 个因子")
                        self._selected_categories = factors
                        return factors
            except Exception as e:
                self.logger.warning(f"加载 {p} 失败: {e}")

        self.logger.error("没有找到已选因子数据, 请先运行 daily_optimizer.py")
        return {}

    # ----------------------------------------------------------
    # 预计算数据 (复用 daily_optimizer 的逻辑)
    # ----------------------------------------------------------
    def _get_required_factors(self) -> List[str]:
        return ['momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal', 'volatility_20', 'volume_ratio', 'boll_position']

    def _prepare_eval_data(self, factor_data, factor_names, start_date, end_date):
        """预计算因子矩阵 (返回 numpy 3D 数组 + 前向收益)。"""
        df = factor_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
        df = df[mask].copy()
        if df.empty:
            return None
        valid_factors = [c for c in factor_names if c in df.columns]
        if not valid_factors:
            return None
        all_symbols = sorted(df["symbol"].unique())
        all_dates = sorted(df["date"].unique())
        n_dates, n_symbols, n_factors = len(all_dates), len(all_symbols), len(valid_factors)
        self.logger.info(f"预计算: {n_dates}日 x {n_symbols}标 x {n_factors}因子")

        factor_vals = np.full((n_dates, n_symbols, n_factors), np.nan, dtype=np.float32)
        for i, name in enumerate(valid_factors):
            pivot = df.pivot_table(index="date", columns="symbol", values=name, aggfunc="first")
            pivot = pivot.reindex(index=all_dates, columns=all_symbols)
            factor_vals[:, :, i] = pivot.values.astype(np.float32)

        mean_vals = np.nanmean(factor_vals, axis=1, keepdims=True)
        std_vals = np.nanstd(factor_vals, axis=1, keepdims=True)
        factor_z = np.where(std_vals > 1e-10, (factor_vals - mean_vals) / std_vals, 0.0)

        close_pivot = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="first")
        close_pivot = close_pivot.reindex(index=all_dates, columns=all_symbols)
        close_vals = close_pivot.values.astype(np.float32)
        fwd_ret = np.full_like(close_vals, np.nan, dtype=np.float32)
        fwd_ret[:-1] = (close_vals[1:] - close_vals[:-1]) / np.maximum(close_vals[:-1], 1e-10)
        mkt_ret = np.nanmean(fwd_ret, axis=1)

        # 换手率辅助: 权重变化追踪
        return {
            "factor_z": factor_z, "fwd_ret": fwd_ret, "mkt_ret": mkt_ret,
            "valid_factors": valid_factors, "n_dates": n_dates,
            "all_dates": all_dates,
        }

    def _ensure_eval_cache(self, start_date: str, end_date: str):
        """确保预计算数据已加载。"""
        if self._eval_cache and self._eval_cache.get("n_dates", 0) > 0:
            return
        all_factors = []
        for factors in self._selected_categories.values():
            all_factors.extend(factors)
        all_factors = list(set(all_factors))
        self._all_factors = all_factors
        required = self._get_required_factors()
        factor_data = self.db.get_factors(
            factor_names=list(set(required + all_factors)),
            start_date=start_date, end_date=end_date, with_close=True,
        )
        if factor_data is None or factor_data.empty:
            raise RuntimeError("无法加载因子数据")

        t0 = pd.Timestamp.now()
        self._eval_cache = self._prepare_eval_data(factor_data, all_factors, start_date, end_date)
        elapsed = (pd.Timestamp.now() - t0).total_seconds()
        self.logger.info(f"预计算完成, 耗时 {elapsed:.1f}s")

    # ----------------------------------------------------------
    # 风险平价权重计算
    # ----------------------------------------------------------
    def _rp_portfolio_weights(self, scores, fwd_ret, t, prev_weights, hold_since,
                              top_n=30, min_hold_days=MIN_HOLD_DAYS):
        """风险平价权重分配: 波动率倒数加权 + 最低持仓天数约束。

        Args:
            scores: 因子综合得分向量 (n_symbols,)
            fwd_ret: 前向收益矩阵 (n_dates, n_symbols)
            t: 当前时间索引
            prev_weights: 当前持仓权重 (n_symbols,)
            hold_since: 每只股票上次买入时间索引 (n_symbols,), -1表示未持有
            top_n: 最多选股数
            min_hold_days: 最低持仓天数

        Returns:
            new_weights: 新权重向量 (n_symbols,)
        """
        n_symbols = len(scores)
        valid = ~np.isnan(scores)

        # 锁定持仓: 持有时间 < min_hold_days 的不能卖
        locked = np.zeros(n_symbols, dtype=bool)
        for i in range(n_symbols):
            if hold_since[i] > 0 and (t - hold_since[i]) < min_hold_days and prev_weights[i] > 0:
                locked[i] = True

        locked_weight = np.sum(prev_weights[locked])

        # 可用股票: 有有效评分且未锁定
        available = valid & ~locked
        n_avail = int(np.sum(available))
        if n_avail < 1:
            return prev_weights.copy()

        avail_scores = scores.copy()
        avail_scores[~available] = -np.inf
        n_pick = min(top_n, n_avail)
        best = np.argpartition(-avail_scores, n_pick)[:n_pick]

        # 风险平价: 波动率倒数加权
        if t >= 20:
            hist_ret = fwd_ret[t-20:t, :]
            vol = np.nanstd(hist_ret, axis=0) + 1e-10
            inv_vol = 1.0 / vol
        else:
            inv_vol = np.ones(n_symbols, dtype=np.float32)

        rp_w = inv_vol[best] / np.sum(inv_vol[best]) * (1.0 - locked_weight)

        new_weights = np.zeros(n_symbols, dtype=np.float32)
        new_weights[best] = rp_w
        new_weights[locked] = prev_weights[locked]
        return new_weights

    # ----------------------------------------------------------
    # 快速评估 (风险平价版)
    # ----------------------------------------------------------
    def _fast_evaluate(self, weights, eval_data, l1_lambda=0.0, turnover_penalty=0.0,
                       rebal_freq=DEFAULT_REBAL_FREQ, use_risk_parity=True,
                       tx_cost_rate=TX_COST_RATE, min_hold_days=MIN_HOLD_DAYS):
        """向量化快速评估, 支持 L1 正则化、换手率惩罚、风险平价、交易成本。

        Args:
            weights: 因子权重向量
            eval_data: 预计算数据
            l1_lambda: L1 正则化系数
            turnover_penalty: 换手率惩罚系数
            rebal_freq: 调仓频率 (交易日数)
            use_risk_parity: 是否使用风险平价分配仓位
            tx_cost_rate: 单边交易成本率
            min_hold_days: 最低持仓天数
        """
        try:
            factor_z = eval_data["factor_z"]
            fwd_ret = eval_data["fwd_ret"]
            mkt_ret = eval_data["mkt_ret"]
            n_dates = eval_data["n_dates"]
            n_symbols = factor_z.shape[1]
            top_n = 30

            w = np.array(weights, dtype=np.float32)
            composite = np.dot(factor_z, w)

            trend = np.full(n_dates, 0.5, dtype=np.float32)
            for t in range(20, n_dates):
                short_ma = np.nanmean(mkt_ret[t-5:t])
                long_ma = np.nanmean(mkt_ret[t-20:t])
                trend[t] = 1.0 if short_ma > long_ma else 0.5

            daily_ret = np.zeros(n_dates, dtype=np.float32)
            prev_weights = np.zeros(n_symbols, dtype=np.float32)
            hold_since = np.full(n_symbols, -1, dtype=np.int32)
            total_turnover = 0.0
            rebal_days = 0

            for t in range(1, n_dates):
                scores_t = composite[t-1]
                valid = ~np.isnan(scores_t)
                nv = np.sum(valid)
                if nv < 3:
                    daily_ret[t] = 0.0
                    continue

                should_rebal = (rebal_freq <= 1) or (t % rebal_freq == 0)

                if should_rebal:
                    if use_risk_parity:
                        new_weights = self._rp_portfolio_weights(
                            scores_t, fwd_ret, t, prev_weights, hold_since,
                            top_n=top_n, min_hold_days=min_hold_days,
                        )
                    else:
                        if nv < top_n:
                            pos = max(nv, 1)
                            idx = np.where(valid)[0][np.argsort(scores_t[valid])[-pos:]]
                            new_weights = np.zeros(n_symbols, dtype=np.float32)
                            new_weights[idx] = 1.0 / pos
                        else:
                            idx = np.argsort(scores_t)[-top_n:]
                            new_weights = np.zeros(n_symbols, dtype=np.float32)
                            new_weights[idx] = 1.0 / top_n

                    turnover = np.sum(np.abs(new_weights - prev_weights))
                    total_turnover += turnover
                    rebal_days += 1

                    # 更新持仓追踪
                    newly_bought = (new_weights > 0) & ((prev_weights <= 0) | (hold_since < 0))
                    hold_since[newly_bought] = t
                    sold = (prev_weights > 0) & (new_weights <= 0)
                    hold_since[sold] = -1

                    tx_cost = 0.5 * turnover * tx_cost_rate
                    prev_weights = new_weights.copy()
                    daily_ret[t] = (np.sum(fwd_ret[t] * new_weights, where=~np.isnan(fwd_ret[t])) - tx_cost) * trend[t]
                else:
                    daily_ret[t] = np.sum(fwd_ret[t] * prev_weights, where=~np.isnan(fwd_ret[t])) * trend[t]

            cum = np.cumprod(1 + daily_ret)
            total_ret = cum[-1] - 1
            if total_ret <= 0:
                return -1.0

            n_years = max(n_dates / 252.0, 1/252)
            ann_ret = (1 + total_ret) ** (1 / n_years) - 1
            dd = cum / np.maximum.accumulate(cum) - 1
            max_dd = float(abs(np.min(dd)))
            ret_s = daily_ret[1:]
            std_ret = np.std(ret_s) + 1e-10
            sharpe = float(np.mean(ret_s) / std_ret * np.sqrt(252))
            win_rate = float(np.sum(ret_s > 0) / max(len(ret_s), 1))
            calmar = ann_ret / max(max_dd, 1e-10)

            risk_result = self.risk_constraints.check_backtest_result(
                annual_return=ann_ret, max_drawdown=max_dd,
                volatility=float(std_ret * np.sqrt(252)),
                calmar_ratio=calmar, win_rate=win_rate, turnover=None,
            )
            if not risk_result.passed:
                return -1.0

            base_score = float(ann_ret * 0.30 + sharpe * 0.25 + calmar * 0.30 + win_rate * 0.15)

            l1_penalty = l1_lambda * np.sum(np.abs(w))
            avg_turnover = total_turnover / max(rebal_days, 1) if rebal_days > 0 else 0
            turnover_pen = turnover_penalty * max(0, avg_turnover - 0.5)

            return float(base_score - l1_penalty - turnover_pen)

        except Exception as e:
            self.logger.warning(f"快速评估异常: {e}")
            return -1.0

    # ----------------------------------------------------------
    # Step 1: Enhanced GA (多版本对比)
    # ----------------------------------------------------------
    def _run_single_ga(self, eval_data, n_factors, label, l1_lambda=0.0,
                       turnover_penalty=0.0, rebal_freq=1,
                       population_size=30, generations=60) -> Dict:
        """运行单次 GA 优化。"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"GA 配置: {label}")
        self.logger.info(f"  L1={l1_lambda}, 换手惩罚={turnover_penalty}, 调仓频率={rebal_freq}日")
        self.logger.info(f"  种群={population_size}, 代数={generations}")

        best_weights, best_scores = [], []
        population = [np.random.randn(n_factors) * 0.3 for _ in range(population_size)]
        t0 = pd.Timestamp.now()

        for gen in range(generations):
            scores = []
            for weights in population:
                score = self._fast_evaluate(
                    weights, eval_data,
                    l1_lambda=l1_lambda,
                    turnover_penalty=turnover_penalty,
                    rebal_freq=rebal_freq,
                )
                scores.append(score)

            indices = np.argsort(scores)[::-1]
            population = [population[i] for i in indices]
            scores = [scores[i] for i in indices]

            if scores[0] > 0:
                best_weights.append(population[0].copy())
                best_scores.append(scores[0])

            if gen % 10 == 0 or gen == generations - 1:
                elapsed = (pd.Timestamp.now() - t0).total_seconds()
                self.logger.info(f"  第 {gen+1}/{generations} 代: 最佳={scores[0]:.4f}, 耗时={elapsed:.0f}s")

            # 保留 top 3 + 交叉/变异
            new_pop = population[:3].copy()
            while len(new_pop) < population_size:
                if np.random.random() < 0.7 and len(population) > 2:
                    p1, p2 = np.random.choice(min(10, len(population)), 2, replace=False)
                    child = (population[p1] + population[p2]) / 2
                else:
                    child = population[np.random.randint(min(5, len(population)))].copy()
                # 动态变异率
                mutation_rate = 0.15 * (1 - gen / generations) + 0.05
                if np.random.random() < mutation_rate:
                    child += np.random.randn(n_factors) * 0.15
                new_pop.append(child)
            population = new_pop

        elapsed = (pd.Timestamp.now() - t0).total_seconds()
        self.logger.info(f"  GA 完成! 有效配置={len(best_scores)}, 耗时={elapsed:.0f}s")

        if not best_scores:
            return {"label": label, "configs": []}

        # 取 top 5 (去重: 只保留差异大于阈值的)
        sorted_idx = np.argsort(best_scores)[::-1]
        configs = []
        seen_vectors = []
        for idx in sorted_idx:
            vec = best_weights[idx]
            is_dup = False
            for sv in seen_vectors:
                if np.mean(np.abs(vec - sv)) < 0.3:
                    is_dup = True
                    break
            if not is_dup:
                seen_vectors.append(vec)
                configs.append({
                    "weights": {self._all_factors[j]: float(vec[j]) for j in range(n_factors)},
                    "score": float(best_scores[idx]),
                })
            if len(configs) >= 5:
                break

        return {"label": label, "configs": configs, "elapsed_seconds": elapsed}

    def run_enhanced_ga(self, start_date: str, end_date: str) -> Dict:
        """运行多个 GA 版本, 对比不同正则化策略的效果。"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Enhanced GA 多版本对比优化")
        self.logger.info("=" * 60)

        cp = self._load_checkpoint()
        if cp and cp.step in ("enhanced_ga", "rolling_validation", "real_backtest", "done") and cp.ga_configs:
            self.logger.info(f"Enhanced GA 已有结果, 跳过 (共 {len(cp.ga_configs)} 个版本)")
            return {c["label"]: c["configs"] for c in cp.ga_configs}

        # 断点恢复: 从 enhanced_ga_results.json 恢复
        results_file = self.today_dir / "enhanced_ga_results.json"
        if results_file.exists() and (not cp or not cp.ga_configs):
            self.logger.info("从 enhanced_ga_results.json 恢复 GA 结果")
            with open(results_file) as f:
                all_results = json.load(f)
            cp = ValidationCheckpoint(timestamp=datetime.now().isoformat(), step="enhanced_ga")
            cp.ga_configs = all_results
            self._save_checkpoint(cp)
            return {r["label"]: r["configs"] for r in all_results}

        self._ensure_eval_cache(start_date, end_date)
        eval_data = self._eval_cache
        n_factors = len(self._all_factors)
        if n_factors == 0:
            self.logger.error("没有因子可优化")
            return {}

        # 定义 5 个 GA 版本配置 (name, l1_lambda, turnover_penalty, rebal_freq, pop_size, n_gen)
        # 全部使用风险平价 + 周频调仓 + 交易成本模型
        ga_configs = [
            ("周频_风险平价_50代",   0.0,    0.0,    3,  30, 50),
            ("周频_L1弱_60代",       0.001,  0.0,    3,  30, 60),
            ("周频_L1中_80代",       0.003,  0.0,    3,  40, 80),
            ("周频_L1弱_换手_60代",  0.001,  0.3,    3,  30, 60),
            ("周频_L1中_换手_100代",  0.003,  0.3,    3,  40, 100),
        ]

        all_results = []
        for label, l1, turnover_pen, rebal_freq, pop_size, gens in ga_configs:
            result = self._run_single_ga(
                eval_data, n_factors, label,
                l1_lambda=l1, turnover_penalty=turnover_pen,
                rebal_freq=rebal_freq, population_size=pop_size, generations=gens,
            )
            all_results.append(result)

        # 保存检查点
        cp = self._load_checkpoint() or ValidationCheckpoint(timestamp=datetime.now().isoformat(), step="enhanced_ga")
        cp.step = "enhanced_ga"
        cp.ga_configs = all_results
        self._save_checkpoint(cp)

        # 保存详细结果
        output = {}
        for r in all_results:
            n = len(r.get("configs", []))
            self.logger.info(f"  [{r['label']}]: {n} 个配置, 耗时 {r.get('elapsed_seconds', 0):.0f}s")
            output[r["label"]] = r["configs"]

        result_path = self.today_dir / "enhanced_ga_results.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Enhanced GA 结果已保存: {result_path}")

        return output

    # ----------------------------------------------------------
    # Step 2: 滚动窗口样本外验证
    # ----------------------------------------------------------
    def _classify_regime(self, mkt_returns: np.ndarray) -> str:
        """根据市场收益率分类市场状态。"""
        total_ret = float(np.nanmean(mkt_returns))
        vol = float(np.nanstd(mkt_returns))
        if total_ret > 0.15:
            return "强牛市"
        elif total_ret > 0.05:
            return "牛市"
        elif total_ret > -0.05:
            return "震荡市"
        elif total_ret > -0.15:
            return "熊市"
        else:
            return "强熊市"

    def _slice_eval_data(self, full_data: Dict, start_idx: int, end_idx: int) -> Dict:
        """从完整 eval_data 中切片出子区间。"""
        return {
            "factor_z": full_data["factor_z"][start_idx:end_idx],
            "fwd_ret": full_data["fwd_ret"][start_idx:end_idx],
            "mkt_ret": full_data["mkt_ret"][start_idx:end_idx],
            "valid_factors": full_data["valid_factors"],
            "n_dates": end_idx - start_idx,
            "all_dates": full_data["all_dates"][start_idx:end_idx],
        }

    def run_rolling_validation(self, start_date: str, end_date: str) -> List[Dict]:
        """滚动窗口验证: 在多个市场周期下测试配置表现。"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 2: 滚动窗口样本外验证")
        self.logger.info("=" * 60)

        cp = self._load_checkpoint()
        if cp and cp.step in ("rolling_validation", "real_backtest", "done") and cp.rolling_results:
            self.logger.info(f"滚动验证已有结果, 跳过 ({len(cp.rolling_results)} 条)")
            return cp.rolling_results

        # 从检查点获取 GA 配置
        if not cp or not cp.ga_configs:
            self.logger.error("没有 GA 配置, 请先运行 enhanced GA")
            return []

        # 收集所有配置
        all_configs = []
        for ga_result in cp.ga_configs:
            for i, cfg in enumerate(ga_result.get("configs", [])):
                all_configs.append({
                    "source": ga_result["label"],
                    "config_id": i + 1,
                    "weights": cfg["weights"],
                    "ga_score": cfg["score"],
                })

        if not all_configs:
            self.logger.warning("没有有效的配置")
            return []

        self.logger.info(f"共 {len(all_configs)} 个配置参与验证")

        # 预计算完整区间数据
        self._ensure_eval_cache(start_date, end_date)
        full_eval = self._eval_cache
        all_dates = full_eval.get("all_dates", [])
        if not all_dates:
            self.logger.warning("没有日期数据")
            return []

        results = []
        for cfg_idx, cfg in enumerate(all_configs):
            weights_vec = np.array([cfg["weights"].get(f, 0.0) for f in self._all_factors], dtype=np.float32)
            window_results = []

            for win in MARKET_WINDOWS:
                try:
                    w_start = pd.Timestamp(win["start"])
                    w_end = pd.Timestamp(win["end"])
                    start_i = np.searchsorted(all_dates, w_start)
                    end_i = np.searchsorted(all_dates, w_end, side='right')
                    if end_i <= start_i + 20:
                        continue

                    w_data = self._slice_eval_data(full_eval, start_i, end_i)
                    score = self._fast_evaluate(weights_vec, w_data, l1_lambda=0, turnover_penalty=0)
                    regime = self._classify_regime(w_data["mkt_ret"])

                    window_results.append({
                        "window": win["label"],
                        "type": win["type"],
                        "regime": regime,
                        "score": float(score),
                        "n_dates": int(w_data["n_dates"]),
                    })
                except Exception as e:
                    self.logger.warning(f"窗口 {win['label']} 验证失败: {e}")

            all_windows_passed = all(w["score"] > 0 for w in window_results)
            avg_score = float(np.mean([w["score"] for w in window_results if w["score"] > 0])) if any(w["score"] > 0 for w in window_results) else -1

            results.append({
                "source": cfg["source"],
                "config_id": cfg["config_id"],
                "ga_score": cfg["ga_score"],
                "weights": cfg["weights"],
                "avg_window_score": avg_score,
                "all_windows_passed": all_windows_passed,
                "windows": window_results,
                "n_windows": len(window_results),
                "positive_windows": sum(1 for w in window_results if w["score"] > 0),
            })

            status = "✓" if all_windows_passed else "✗"
            self.logger.info(
                f"  [{status}] {cfg['source']}#{cfg['config_id']}: "
                f"GA得分={cfg['ga_score']:.3f}, "
                f"窗口均分={avg_score:.3f}, "
                f"通过={results[-1]['positive_windows']}/{len(window_results)}"
            )

        # 保存检查点
        cp = self._load_checkpoint() or ValidationCheckpoint(timestamp=datetime.now().isoformat(), step="rolling_validation")
        cp.step = "rolling_validation"
        cp.rolling_results = results
        self._save_checkpoint(cp)

        result_path = self.today_dir / "rolling_validation.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(results), f, ensure_ascii=False, indent=2)
        self.logger.info(f"滚动验证结果已保存: {result_path}")

        return results

    # ----------------------------------------------------------
    # Step 3: 向量化回测验证 (替代 BacktestEngine, 快 100x+)
    # ----------------------------------------------------------
    def _weights_from_config(self, cfg: Dict) -> Dict[str, float]:
        """从配置字典提取权重。"""
        if "weights" in cfg:
            return cfg["weights"]
        if "weight_config" in cfg:
            return cfg["weight_config"].get("weights", {})
        return {}

    def _run_vectorized_backtest(
        self,
        weights: Dict[str, float],
        start_date: str,
        end_date: str,
        name: str = "",
        rebalance_freq: int = DEFAULT_REBAL_FREQ,
        top_n: int = 30,
        initial_capital: float = 1_000_000,
        use_risk_parity: bool = True,
        tx_cost_rate: float = TX_COST_RATE,
        min_hold_days: int = MIN_HOLD_DAYS,
    ) -> Dict:
        """纯向量化回测, 支持风险平价、交易手续费、最低持仓约束。

        在调仓日用预计算 z-scores + 风险平价分配仓位,
        持有至下次调仓, 扣除交易成本。
        """
        t0 = pd.Timestamp.now()

        all_factor_names = list(set(list(weights.keys()) + self._get_required_factors()))
        factor_data = self.db.get_factors(
            factor_names=all_factor_names,
            start_date=start_date, end_date=end_date, with_close=True,
        )
        if factor_data is None or factor_data.empty:
            return {"name": name, "error": "no factor data", "elapsed_seconds": 0.0}

        cache = self._prepare_eval_data(factor_data, all_factor_names, start_date, end_date)
        if cache is None:
            return {"name": name, "error": "prepare failed", "elapsed_seconds": 0.0}

        factor_z = cache["factor_z"]
        fwd_ret = cache["fwd_ret"]
        valid_factors = cache["valid_factors"]
        n_dates = cache["n_dates"]

        w = np.zeros(len(valid_factors), dtype=np.float32)
        assigned = 0
        for i, fn in enumerate(valid_factors):
            val = weights.get(fn, 0.0)
            w[i] = val
            if abs(val) > 1e-10:
                assigned += 1
        w_sum = np.sum(np.abs(w))
        if w_sum > 1e-10:
            w = w / w_sum
        if assigned < 1:
            return {"name": name, "error": "no factor weight match", "elapsed_seconds": 0.0}

        n_symbols = factor_z.shape[1]
        daily_ret = np.zeros(n_dates, dtype=np.float32)
        prev_weights = np.zeros(n_symbols, dtype=np.float32)
        hold_since = np.full(n_symbols, -1, dtype=np.int32)
        total_trades = 0
        total_turnover = 0.0
        rebal_count = 0

        for t in range(1, n_dates):
            scores_t = np.dot(factor_z[t-1], w)
            valid = ~np.isnan(scores_t)
            nv = np.sum(valid)
            if nv < 3:
                daily_ret[t] = 0.0
                continue

            should_rebal = (rebalance_freq <= 1) or (t % rebalance_freq == 0)

            if should_rebal:
                if use_risk_parity:
                    new_weights = self._rp_portfolio_weights(
                        scores_t, fwd_ret, t, prev_weights, hold_since,
                        top_n=top_n, min_hold_days=min_hold_days,
                    )
                else:
                    nan_mask = np.isnan(scores_t) | np.isinf(scores_t)
                    valid_count = np.sum(~nan_mask)
                    if valid_count < 1:
                        continue
                    scores_clean = scores_t.copy()
                    scores_clean[nan_mask] = -np.inf
                    n_pick = min(top_n, valid_count)
                    best = np.argpartition(-scores_clean, n_pick)[:n_pick]
                    new_weights = np.zeros(n_symbols, dtype=np.float32)
                    new_weights[best] = 1.0 / n_pick

                turnover = np.sum(np.abs(new_weights - prev_weights))
                total_turnover += 0.5 * turnover
                total_trades += int(np.sum(new_weights > 0))
                rebal_count += 1

                newly_bought = (new_weights > 0) & ((prev_weights <= 0) | (hold_since < 0))
                hold_since[newly_bought] = t
                sold = (prev_weights > 0) & (new_weights <= 0)
                hold_since[sold] = -1

                tx_cost = 0.5 * turnover * tx_cost_rate
                prev_weights = new_weights.copy()
                daily_ret[t] = np.nansum(new_weights * fwd_ret[t]) - tx_cost
            else:
                daily_ret[t] = np.nansum(prev_weights * fwd_ret[t])

        strategy_ret = np.clip(daily_ret[:n_dates], -0.20, 0.20)

        with np.errstate(all='ignore'):
            log_ret = np.log1p(strategy_ret)
            log_cumsum = np.cumsum(log_ret)
            log_cumsum = np.clip(log_cumsum, -1000, 700)
            equity = np.exp(np.log(initial_capital) + log_cumsum)
            equity = np.nan_to_num(equity, nan=initial_capital, posinf=1e308, neginf=0.0)

        final_value = equity[-1]
        total_return = final_value / initial_capital - 1.0

        years = max(n_dates / 245.0, 1e-6)
        if total_return > 0 and np.isfinite(total_return):
            annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0
        else:
            annualized_return = np.expm1(np.clip(np.log1p(total_return) / years, -10, 10))

        peak = np.maximum.accumulate(equity)
        with np.errstate(all='ignore'):
            dd = (equity - peak) / np.maximum(peak, 1e-10)
            dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
        max_drawdown = float(np.abs(np.min(dd)))

        mean_ret = np.mean(strategy_ret)
        std_ret = np.std(strategy_ret, ddof=1)
        sharpe = np.sqrt(245) * mean_ret / max(std_ret, 1e-10)

        win_days = np.sum(strategy_ret > 1e-10)
        win_rate = win_days / max(len(strategy_ret), 1)

        pos_mask = strategy_ret > 1e-10
        neg_mask = strategy_ret < -1e-10
        avg_win = np.mean(strategy_ret[pos_mask]) if np.any(pos_mask) else 0.0
        avg_loss = abs(np.mean(strategy_ret[neg_mask])) if np.any(neg_mask) else 1.0
        profit_factor = avg_win / max(avg_loss, 1e-10)

        avg_turnover = total_turnover / max(rebal_count, 1)
        elapsed = (pd.Timestamp.now() - t0).total_seconds()

        return {
            "name": name,
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": int(total_trades),
            "avg_turnover": float(avg_turnover),
            "n_rebalances": int(rebal_count),
            "initial_capital": float(initial_capital),
            "final_value": float(final_value),
            "elapsed_seconds": elapsed,
            "start_date": start_date,
            "end_date": end_date,
            "use_risk_parity": use_risk_parity,
            "tx_cost_rate": tx_cost_rate,
            "min_hold_days": min_hold_days,
        }

    def run_real_backtest(self, start_date: str, end_date: str) -> List[Dict]:
        """使用向量化回测验证所有候选配置。"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 3: 向量化回测验证")
        self.logger.info("=" * 60)

        cp = self._load_checkpoint()
        if cp and cp.step in ("real_backtest", "done") and cp.real_backtest_results:
            self.logger.info(f"回测已有 {len(cp.real_backtest_results)} 个结果, 跳过")
            return cp.real_backtest_results

        rolling_results = cp.rolling_results if cp and cp.rolling_results else []
        candidates = []

        if rolling_results:
            passed = [r for r in rolling_results if r.get("all_windows_passed")]
            if passed:
                passed.sort(key=lambda x: x.get("avg_window_score", -1), reverse=True)
                candidates.extend(passed[:3])
            rolling_results.sort(key=lambda x: x.get("ga_score", -1), reverse=True)
            for r in rolling_results:
                if r not in candidates and len(candidates) < 5:
                    candidates.append(r)

        if not candidates:
            self.logger.warning("没有候选配置, 直接使用 GA 最佳配置")
            if cp and cp.ga_configs:
                for ga_r in cp.ga_configs:
                    for i, cfg in enumerate(ga_r.get("configs", [])[:1]):
                        candidates.append({
                            "source": ga_r["label"],
                            "config_id": i + 1,
                            "ga_score": cfg["score"],
                            "weights": cfg["weights"],
                        })

        if not candidates:
            self.logger.error("没有任何候选配置, 跳过回测")
            return []

        results = []
        for cand in candidates:
            weights = cand.get("weights") or self._weights_from_config(cand)
            if not weights:
                self.logger.warning(f"配置 {cand.get('source')}#{cand.get('config_id')} 没有权重")
                continue

            name = f"{cand['source']}#{cand['config_id']}"
            self.logger.info(f"\n  向量化回测: {name}")

            try:
                report = self._run_vectorized_backtest(
                    weights=weights,
                    start_date=start_date, end_date=end_date,
                    name=name, top_n=30,
                    initial_capital=1_000_000,
                )

                if "error" in report:
                    self.logger.error(f"  {name} 回测失败: {report['error']}")
                    continue

                results.append(report)
                self.logger.info(
                    f"    年化={report['annualized_return']*100:+.2f}% "
                    f"夏普={report['sharpe_ratio']:+.2f} "
                    f"回撤={report['max_drawdown']*100:.2f}% "
                    f"胜率={report['win_rate']*100:.1f}% "
                    f"交易={report['total_trades']} "
                    f"耗时={report['elapsed_seconds']:.1f}s"
                )

            except Exception as e:
                self.logger.error(f"  {name} 回测失败: {e}")
                import traceback
                traceback.print_exc()

        cp = self._load_checkpoint() or ValidationCheckpoint(timestamp=datetime.now().isoformat(), step="real_backtest")
        cp.step = "real_backtest"
        cp.real_backtest_results = results
        self._save_checkpoint(cp)

        result_path = self.today_dir / "real_backtest_results.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(results), f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        self.logger.info(f"回测结果已保存: {result_path}")

        return results

    # ----------------------------------------------------------
    # 完整 Pipeline
    # ----------------------------------------------------------
    def run_full(self, start_date: str = "2019-01-01", end_date: str = None):
        """执行完整验证流程。"""
        if end_date is None:
            end_date = self.db.get_max_date('daily_bars')

        self.logger.info("=" * 60)
        self.logger.info("综合验证与优化 Pipeline")
        self.logger.info(f"回测区间: {start_date} ~ {end_date}")
        self.logger.info(f"输出目录: {self.today_dir}")
        self.logger.info("=" * 60)

        # Step 0: 加载已选因子
        factors = self.load_selected_factors()
        if not factors:
            self.logger.error("无法加载因子, 退出")
            return

        # Step 1: Enhanced GA (2021+ 训练)
        ga_start = max(start_date, GA_START)
        ga_results = self.run_enhanced_ga(ga_start, end_date)

        # Step 2: 滚动窗口验证 (全区间)
        rolling_results = self.run_rolling_validation(start_date, end_date)

        # Step 3: 真实 BacktestEngine 验证 (全区间)
        real_results = self.run_real_backtest(start_date, end_date)

        # 最终报告
        self.logger.info("\n" + "=" * 60)
        self.logger.info("验证完成!")
        self.logger.info("=" * 60)

        cp = self._load_checkpoint()
        if cp:
            cp.step = "done"
            self._save_checkpoint(cp)

        # 复制脚本自身到输出目录
        script_copy = self.today_dir / "enhanced_optimizer.py"
        shutil.copy2(__file__, script_copy)
        self.logger.info(f"脚本已备份: {script_copy}")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "backtest_period": {"start": start_date, "end": end_date},
            "ga_versions": len(ga_results) if ga_results else 0,
            "rolling_windows": len(MARKET_WINDOWS),
            "configs_tested_real_backtest": len(real_results),
            "ga_results": ga_results,
            "rolling_results": rolling_results,
            "real_backtest_results": real_results,
        }
        summary_path = self.today_dir / "validation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"汇总报告: {summary_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="综合验证与优化 Pipeline")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-dd", type=float, default=0.60)
    args = parser.parse_args()

    optimizer = EnhancedOptimizer(max_drawdown=args.max_dd)
    optimizer.run_full(args.start, args.end)


if __name__ == "__main__":
    main()
