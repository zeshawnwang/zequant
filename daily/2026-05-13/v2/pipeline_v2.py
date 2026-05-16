import os, sys, json, time, logging, copy, math, gc
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core.database import Database

def find_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    while current and current != '/':
        if os.path.isdir(os.path.join(current, 'core')):
            return current
        current = os.path.dirname(current)
    return script_dir

PROJ_ROOT = find_project_root()
RUN_DAY = "2026-05-13"
VERSION = "v2"
OUT_DIR = os.path.join(PROJ_ROOT, "daily", RUN_DAY, VERSION)
LOG_DIR = OUT_DIR

LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline_v2")

logger.info("=" * 60)
logger.info("V2 Pipeline 启动")
logger.info(f"项目根目录: {PROJ_ROOT}")
logger.info(f"输出目录: {OUT_DIR}")
logger.info("=" * 60)

TX_COST_RATE = 0.0012

PENALTY_VALUES = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]

ALLOCATOR_CONFIGS = {
    "原始风险平价": {"use_hysteresis": False},
    "迟滞_默认": {
        "use_hysteresis": True,
        "hysteresis_kwargs": {
            "large_pos_threshold": 0.10,
            "min_adjust_delta": 0.02,
            "keep_ratio": 0.70
        }
    },
    "迟滞_激进": {
        "use_hysteresis": True,
        "hysteresis_kwargs": {
            "large_pos_threshold": 0.05,
            "min_adjust_delta": 0.01,
            "keep_ratio": 0.50
        }
    },
}

EXPERIMENTS = []
for penalty in PENALTY_VALUES:
    for alloc_name, alloc_cfg in ALLOCATOR_CONFIGS.items():
        EXPERIMENTS.append({
            "name": f"pen{penalty}_{alloc_name}",
            "l1_coef": 0.003,
            "turnover_penalty": penalty,
            "top_n": 40,
            "rebal_freq": 3,
            "ngen": 30,
            "npop": 15,
            "mutate_prob": 0.15,
            "crossover_prob": 0.7,
            "stall_limit": 20,
            **alloc_cfg,
        })

NUM_EXPERIMENTS = len(EXPERIMENTS)
logger.info(f"实验设计: {NUM_EXPERIMENTS}组 (6换手惩罚 × 3分配器类型)")
for i, exp in enumerate(EXPERIMENTS):
    hs = "迟滞" if exp["use_hysteresis"] else "原始"
    logger.info(f"  [{i+1}/{NUM_EXPERIMENTS}] {exp['name']} | L1={exp['l1_coef']} ngen={exp['ngen']}")

CHECKPOINT_FILE = os.path.join(OUT_DIR, "checkpoint.json")
BEST_CONFIG_FILE = os.path.join(OUT_DIR, "best_config.json")
GA_RESULTS_FILE = os.path.join(OUT_DIR, "ga_results.json")
BACKTEST_RESULTS_FILE = os.path.join(OUT_DIR, "backtest_results.json")
BACKTEST_REPORT_FILE = os.path.join(OUT_DIR, "backtest_report.html")
EQUITY_CURVE_FILE = os.path.join(OUT_DIR, "best_equity_curve.json")

L1 = 0.003
TOP_N = 40
REBAL_FREQ = 3
N_FACTORS = None

os.makedirs(OUT_DIR, exist_ok=True)

V1_FACTOR_NAMES = ['a27', 'a30', 'a31', 'a41', 'a42', 'a64', 'a69', 'a8', 'a80', 'a85',
    'a88', 'a91', 'a97', 'a98', 'a99', 'ff_mkt', 'gtja103', 'gtja104', 'gtja105',
    'gtja108', 'gtja113', 'gtja117', 'gtja12', 'gtja120', 'gtja121', 'gtja123',
    'gtja127', 'gtja13', 'gtja139', 'gtja141', 'gtja142', 'gtja144', 'gtja148',
    'gtja164', 'gtja168', 'gtja171', 'gtja176', 'gtja185', 'gtja34', 'gtja49',
    'gtja62', 'gtja76', 'gtja83', 'gtja85', 'gtja90', 'gtja91', 'gtja99',
    'returns', 'rsi_14', 'volatility_20']

factor_name_map = {
    'a91': '市净率倒数', 'a8': '换手率_20日均值', 'a27': '成交量_20日均值',
    'a41': '日内振幅_20日均值', 'a64': '动量_20日', 'a69': '波动率_20日',
    'a80': '乖离率_20日', 'a85': '资金流向_20日均值', 'a30': '振幅_30日',
    'a31': '换手率_30日均值', 'a42': '日内振幅_30日均值', 'a88': 'RSI_14日',
    'a97': '动量_60日', 'a98': '波动率_60日', 'a99': '乖离率_60日',
    'returns': '收益率', 'volatility_20': '20日波动率', 'rsi_14': '14日RSI',
    'ff_mkt': '市场因子',
}

def get_factor_names(columns):
    names = []
    for col in columns:
        if col in factor_name_map:
            names.append(f"{factor_name_map[col]}({col})")
        else:
            names.append(str(col))
    return names

V1_BEST_WEIGHTS = None
V1_WEIGHTS_PATH = os.path.join(OUT_DIR, 'v1_reference', 'ga_results.json')
if os.path.exists(V1_WEIGHTS_PATH):
    try:
        with open(V1_WEIGHTS_PATH) as f:
            v1_data = json.load(f)
        for item in v1_data:
            if 'L1中_80代' in item['label']:
                w = item['configs'][0]['weights']
                V1_BEST_WEIGHTS = np.array([w.get(f, 0.0) for f in V1_FACTOR_NAMES], dtype=np.float32)
                logger.info(f"已加载V1最佳权重 ({len(V1_BEST_WEIGHTS)}个因子, "
                           f"{int(np.sum(np.abs(V1_BEST_WEIGHTS) > 1e-6))}个非零)")
                break
    except Exception as e:
        logger.warning(f"V1权重加载失败: {e}")

def load_data():
    global N_FACTORS
    logger.info("加载数据...")
    db = Database()
    
    symbols_df = db.get_symbols()
    ticker_list = symbols_df['symbol'].tolist()
    logger.info(f"股票池: {len(ticker_list)}只")
    
    factor_df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                               factor_names=V1_FACTOR_NAMES, with_close=True)
    logger.info(f"因子数据: {factor_df.shape}")
    
    meta_cols = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change'}
    factor_cols = [c for c in V1_FACTOR_NAMES if c in factor_df.columns]
    logger.info(f"V1因子: {len(factor_cols)}/{len(V1_FACTOR_NAMES)}个")
    N_FACTORS = len(factor_cols)
    
    factor_names = get_factor_names(factor_cols)
    
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    all_dates = sorted(factor_df['date'].unique())
    dates_arr = np.array(all_dates, dtype='datetime64')
    logger.info(f"日期范围: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)}天)")
    
    n_dates = len(all_dates)
    n_symbols = len(ticker_list)
    n_factors = len(factor_cols)
    
    ticker_to_idx = {t: i for i, t in enumerate(ticker_list)}
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    score_3d = np.zeros((n_dates, n_symbols, n_factors), dtype=np.float32)
    data_mask = np.zeros((n_dates, n_symbols), dtype=bool)
    
    logger.info("构建因子数组 (np极简)...")
    di_map = np.array([date_to_idx[d] for d in factor_df['date']], dtype=np.int32)
    si_map = np.array([ticker_to_idx.get(s, -1) for s in factor_df['symbol']], dtype=np.int32)
    valid = si_map >= 0
    di_map = di_map[valid]
    si_map = si_map[valid]
    
    close_arr = np.zeros((n_dates, n_symbols), dtype=np.float32)
    for fi, fc in enumerate(factor_cols):
        vals = factor_df[fc].values[valid].astype(np.float32)
        np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        score_3d[di_map, si_map, fi] = vals
        if fi == 0:
            data_mask[di_map, si_map] = True
    
    close_vals = factor_df['close'].values[valid].astype(np.float32)
    np.nan_to_num(close_vals, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    close_arr[di_map, si_map] = close_vals
    
    score_3d = np.nan_to_num(score_3d, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info("计算收益率...")
    fwd_rets = np.zeros((n_dates, n_symbols), dtype=np.float32)
    mask_nonzero = close_arr > 1e-10
    for di in range(n_dates - 1):
        curr_valid = mask_nonzero[di]
        nxt_valid = mask_nonzero[di + 1]
        both = curr_valid & nxt_valid
        fwd_rets[di, both] = (close_arr[di + 1, both] - close_arr[di, both]) / close_arr[di, both]
    fwd_rets = np.nan_to_num(fwd_rets, nan=0.0, posinf=0.0, neginf=0.0)
    
    has_return_data = np.any(np.abs(fwd_rets) > 1e-10, axis=1)
    final_mask = data_mask & has_return_data[:, np.newaxis]
    
    logger.info(f"数据加载完成: {score_3d.shape[0]}天 × {score_3d.shape[1]}只 × {score_3d.shape[2]}因子")
    logger.info(f"有收益数据的日期比例: {np.mean(has_return_data)*100:.1f}%")
    logger.info(f"有数据的格子比例: {np.mean(final_mask)*100:.1f}%")
    
    return score_3d, fwd_rets, ticker_list, factor_cols, dates_arr, final_mask, factor_names, db

def compute_rp_weights(scores, fwd_ret, ret_history, prev_weights, hold_since,
                       top_n=30, min_hold_days=5):
    n_symbols = len(scores)
    new_weights = np.zeros(n_symbols, dtype=np.float32)
    
    sorted_idx = np.argsort(-scores)
    top_idx = sorted_idx[:top_n]
    
    valid = np.zeros(n_symbols, dtype=bool)
    valid[top_idx] = True
    
    locked = np.zeros(n_symbols, dtype=bool)
    for i in range(n_symbols):
        if hold_since[i] > 0 and prev_weights[i] > 0 and (ret_history - hold_since[i]) < min_hold_days:
            locked[i] = True
    
    available = valid & ~locked
    avail_idx = np.where(available)[0]
    
    if len(avail_idx) == 0:
        avail_idx = np.where(valid)[0]
    
    locked_weight = float(np.sum(prev_weights[locked]))
    remaining = 1.0 - locked_weight
    
    if remaining <= 0.0:
        return prev_weights.copy()
    
    if fwd_ret.shape[0] >= 20:
        lookback = min(20, fwd_ret.shape[0])
        hist_ret = fwd_ret[-lookback:, :]
        vol = np.nanstd(hist_ret, axis=0) + 1e-10
        vol = np.nan_to_num(vol, nan=1e10, posinf=1e10, neginf=1e10)
    else:
        vol = np.ones(n_symbols, dtype=np.float32)
    
    inv_vol = 1.0 / vol[avail_idx]
    inv_vol_sum = float(np.sum(inv_vol))
    if inv_vol_sum > 0:
        target_weights = (inv_vol / inv_vol_sum) * remaining
    else:
        target_weights = np.ones(len(avail_idx), dtype=np.float32) * remaining / len(avail_idx)
    
    new_weights[locked] = prev_weights[locked]
    new_weights[avail_idx] = target_weights
    
    return new_weights

def make_hysteresis_adjustment(target_weights, prev_weights, valid_mask,
                                large_pos_threshold=0.10, min_adjust_delta=0.02,
                                keep_ratio=0.70):
    delta = target_weights - prev_weights
    delta_thresholded = delta.copy()
    
    large_pos_mask = prev_weights >= large_pos_threshold
    small_adjust = np.abs(delta) < min_adjust_delta
    skip_large_pos = large_pos_mask & small_adjust
    delta_thresholded[skip_large_pos] = 0.0
    
    adjust_idx = np.where((np.abs(delta_thresholded) > 1e-8) & valid_mask)[0]
    if len(adjust_idx) > 0:
        abs_deltas = np.abs(delta_thresholded[adjust_idx])
        order = np.argsort(-abs_deltas)
        n_keep = max(1, int(len(order) * keep_ratio))
        discard_idx = adjust_idx[order[n_keep:]]
        delta_thresholded[discard_idx] = 0.0
    
    adjusted = prev_weights.copy()
    adjusted = adjusted.astype(np.float64)
    adjusted += delta_thresholded.astype(np.float64)
    adjusted = np.maximum(adjusted, 0.0)
    total = float(np.sum(adjusted))
    if total > 0:
        adjusted = adjusted / total
    else:
        adjusted = np.zeros_like(adjusted)
        if len(valid_mask) > 0:
            adjusted[valid_mask] = 1.0 / np.sum(valid_mask)
    
    return adjusted.astype(np.float32)

def ga_fast_evaluate(combined_scores, fwd_rets, t_start, t_end, data_mask,
                     l1_coef, turnover_penalty, top_n=30, rebalance_freq=3,
                     min_hold_days=5, use_hysteresis=False, hysteresis_kwargs=None):
    n_dates = t_end - t_start
    n_symbols = fwd_rets.shape[1]
    
    prev_w = np.zeros(n_symbols, dtype=np.float32)
    hold_since = np.full(n_symbols, -1, dtype=np.int32)
    ret_history_len = 0
    
    total_ret = 0.0
    n_trades = 0
    trade_days = []
    
    for i in range(1, n_dates):
        t = t_start + i
        scores_t = combined_scores[t]
        day_data_ok = data_mask[t]
        
        rebalance = (i % rebalance_freq == 0)
        if not rebalance:
            mask_ok = day_data_ok & (prev_w > 0)
            if np.any(mask_ok):
                p = prev_w[mask_ok].copy()
                p_sum = float(np.sum(p))
                if p_sum > 0:
                    p = p / p_sum
                prev_w = np.zeros(n_symbols, dtype=np.float32)
                prev_w[mask_ok] = p
            if not np.isnan(fwd_rets[t]).any() and not np.isinf(fwd_rets[t]).any():
                daily_ret = float(np.dot(prev_w, fwd_rets[t]))
                if np.isnan(daily_ret) or np.isinf(daily_ret):
                    daily_ret = 0.0
                tx_cost_daily = 0.0
                daily_ret -= tx_cost_daily
                total_ret += daily_ret
                ret_history_len += 1
            continue
        
        if use_hysteresis:
            target = compute_rp_weights(
                scores_t, fwd_rets[:t+1], ret_history_len,
                prev_w, hold_since,
                top_n=top_n, min_hold_days=min_hold_days
            )
            hk = hysteresis_kwargs or {}
            new_w = make_hysteresis_adjustment(
                target, prev_w, day_data_ok,
                large_pos_threshold=hk.get("large_pos_threshold", 0.10),
                min_adjust_delta=hk.get("min_adjust_delta", 0.02),
                keep_ratio=hk.get("keep_ratio", 0.70)
            )
        else:
            clean_scores = np.nan_to_num(scores_t, nan=-1e10, neginf=-1e10)
            new_w = compute_rp_weights(
                clean_scores, fwd_rets[:t+1], ret_history_len,
                prev_w, hold_since,
                top_n=top_n, min_hold_days=min_hold_days
            )
        
        turnover = float(np.sum(np.abs(new_w - prev_w)))
        n_trades += 1
        trade_days.append(t)
        
        tx_cost = 0.5 * turnover * TX_COST_RATE
        daily_ret = float(np.dot(new_w, fwd_rets[t]))
        if np.isnan(daily_ret) or np.isinf(daily_ret):
            daily_ret = 0.0
        daily_ret -= tx_cost
        total_ret += daily_ret
        ret_history_len += 1
        
        prev_w = new_w
        for sym in range(n_symbols):
            if new_w[sym] > 0 and hold_since[sym] < 0:
                hold_since[sym] = ret_history_len
    
    n_trades_val = max(n_trades, 1)
    
    l1_reg = l1_coef * float(np.sum(np.abs(combined_scores.mean(axis=0))))
    score = np.log1p(max(total_ret, 0.0)) / np.sqrt(n_trades_val)
    score -= l1_reg
    
    turnover_pen = turnover_penalty * (n_trades_val / (n_dates / rebalance_freq))
    score -= turnover_pen
    
    if np.isnan(score) or np.isinf(score):
        score = 0.0
    
    return score

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                cp = json.load(f)
            logger.info(f"发现检查点: {len(cp.get('ga_results', []))}/{NUM_EXPERIMENTS} 组已完成")
            return cp
        except Exception as e:
            logger.warning(f"检查点读取失败: {e}")
    return {"ga_results": [], "best_overall": None}

def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    logger.debug(f"检查点已保存 ({len(checkpoint['ga_results'])}/{NUM_EXPERIMENTS})")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

def run_single_ga(score_3d, fwd_rets, data_mask, train_idx, val_idx, config):
    name = config["name"]
    ngen = config["ngen"]
    npop = config["npop"]
    l1_coef = config["l1_coef"]
    turnover_penalty = config["turnover_penalty"]
    top_n = config["top_n"]
    rebal_freq = config["rebal_freq"]
    mutate_prob = config["mutate_prob"]
    crossover_prob = config["crossover_prob"]
    stall_limit = config["stall_limit"]
    use_hysteresis = config.get("use_hysteresis", False)
    hysteresis_kwargs = config.get("hysteresis_kwargs", None)
    
    start_time = time.time()
    
    n_factors = score_3d.shape[2]
    n_symbols = score_3d.shape[1]
    
    np.random.seed(42)
    
    if V1_BEST_WEIGHTS is not None:
        noise = np.random.normal(0, 0.05, (npop, n_factors)).astype(np.float32)
        pop = V1_BEST_WEIGHTS[np.newaxis, :] + noise
        pop = np.clip(pop, -0.5, 0.5).astype(np.float32)
        logger.info(f"  [V2] 使用V1最佳权重初始化GA种群 (均值噪声0.05)")
    else:
        pop = np.random.uniform(-0.5, 0.5, (npop, n_factors)).astype(np.float32)
    
    fitness = np.full(npop, -np.inf, dtype=np.float32)
    best_fitness = -np.inf
    best_weights = None
    stall_count = 0
    gen_fitnesses = []

    train_slice = score_3d[train_idx]
    n_train = train_idx.stop - train_idx.start
    sample_step = 3
    sampled_idx = np.arange(0, n_train, sample_step)
    train_sample = train_slice[sampled_idx]
    fwd_sample = fwd_rets[train_idx.start + sampled_idx]
    mask_sample = data_mask[train_idx.start + sampled_idx]
    logger.info(f"  GA训练数据降采样: {n_train}天 → {len(sampled_idx)}天 (每{sample_step}天取1)")
    
    for gen in range(ngen):
        gen_time = time.time()
        
        for i in range(npop):
            if not np.isfinite(fitness[i]):
                fw = np.zeros(n_factors, dtype=np.float32)
                fw[:] = pop[i]
                combined = np.tensordot(train_sample, fw, axes=(2, 0))
                combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
                s = ga_fast_evaluate(
                    combined, fwd_sample, 0, len(sampled_idx), mask_sample,
                    l1_coef, turnover_penalty,
                    top_n=top_n, rebalance_freq=rebal_freq,
                    use_hysteresis=use_hysteresis, hysteresis_kwargs=hysteresis_kwargs
                )
                if np.isfinite(s):
                    fitness[i] = s
                else:
                    fitness[i] = 0.0
        
        curr_best = float(np.max(fitness))
        curr_best_idx = int(np.argmax(fitness))
        gen_fitnesses.append(curr_best)
        
        if curr_best > best_fitness:
            best_fitness = curr_best
            best_weights = pop[curr_best_idx].copy()
            stall_count = 0
        else:
            stall_count += 1
        
        gen_elapsed = time.time() - gen_time
        mean_fit = float(np.mean(fitness[np.isfinite(fitness)]))
        logger.info(f"  [{name}] 第{gen+1:3d}/{ngen}代 | best={curr_best:.4f} avg={mean_fit:.4f} "
                    f"best_ever={best_fitness:.4f} stall={stall_count} [{gen_elapsed:.1f}s]")
        
        if stall_count >= stall_limit:
            logger.info(f"  [{name}] 早停于第{gen+1}代")
            for j in range(gen+1, ngen):
                gen_fitnesses.append(best_fitness)
            break
        
        if gen == ngen - 1:
            break
        
        new_pop = []
        elite_idx = np.argsort(-fitness)[:max(2, npop // 10)]
        elite_fitnesses = {}
        for idx in elite_idx:
            if np.isfinite(fitness[idx]):
                new_pop.append(pop[idx].copy())
                elite_fitnesses[len(new_pop) - 1] = fitness[idx]
        
        while len(new_pop) < npop:
            if np.random.random() < crossover_prob:
                i1, i2 = np.random.randint(0, npop, 2)
                p1, p2 = pop[i1].copy(), pop[i2].copy()
                alpha = np.random.random(n_factors).astype(np.float32)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
                new_pop.append(c1)
                if len(new_pop) < npop:
                    new_pop.append(c2)
            else:
                i = np.random.randint(0, npop)
                new_pop.append(pop[i].copy())
        
        new_pop = new_pop[:npop]
        new_pop_arr = np.array(new_pop, dtype=np.float32)
        
        mutation_mask = np.random.random(new_pop_arr.shape) < mutate_prob
        mutation = np.random.normal(0, 0.1, new_pop_arr.shape).astype(np.float32)
        new_pop_arr[mutation_mask] += mutation[mutation_mask]
        new_pop_arr = np.clip(new_pop_arr, -0.5, 0.5)
        
        new_fitness = np.full(npop, -np.inf, dtype=np.float32)
        for ei, fv in elite_fitnesses.items():
            new_fitness[ei] = fv
        fitness = new_fitness
        
        pop = new_pop_arr
    
    elapsed = time.time() - start_time
    
    pop_fitness = fitness.copy()
    for i in range(npop):
        if not np.isfinite(pop_fitness[i]):
            fw = np.zeros(n_factors, dtype=np.float32)
            fw[:] = pop[i]
            combined = np.tensordot(train_slice, fw, axes=(2, 0))
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
            s = ga_fast_evaluate(
                combined, fwd_rets, train_idx.start, train_idx.stop, data_mask,
                l1_coef, turnover_penalty,
                top_n=top_n, rebalance_freq=rebal_freq,
                use_hysteresis=use_hysteresis, hysteresis_kwargs=hysteresis_kwargs
            )
            pop_fitness[i] = s if np.isfinite(s) else 0.0
    
    logger.info(f"  [{name}] 完成! best={best_fitness:.4f} 耗时{elapsed:.1f}s")
    
    return {
        "name": name,
        "best_fitness": float(best_fitness),
        "best_weights": best_weights.tolist() if best_weights is not None else None,
        "gen_fitnesses": gen_fitnesses,
        "n_generations": len(gen_fitnesses),
        "elapsed": elapsed,
        "config": {
            "l1_coef": l1_coef,
            "turnover_penalty": turnover_penalty,
            "top_n": top_n,
            "rebal_freq": rebal_freq,
            "ngen": ngen,
            "use_hysteresis": use_hysteresis,
            "hysteresis_kwargs": hysteresis_kwargs,
        }
    }

def rolling_validate(score_3d, fwd_rets, data_mask, tickers, columns, config, factor_weights):
    logger.info(f"滚动回测: {config['name']}")
    
    n_total = score_3d.shape[0]
    train_end = int(n_total * 0.7)
    val_end = int(n_total * 0.85)
    
    val_start = train_end
    test_start = val_end
    
    n_factors = score_3d.shape[2]
    fw = np.zeros(n_factors, dtype=np.float32)
    if factor_weights is not None:
        fw[:] = np.array(factor_weights, dtype=np.float32)
    
    val_combined = np.tensordot(score_3d[val_start:val_end], fw, axes=(2, 0))
    val_combined = np.nan_to_num(val_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_score = ga_fast_evaluate(
        val_combined, fwd_rets, val_start, val_end, data_mask,
        config["l1_coef"], config["turnover_penalty"],
        top_n=config["top_n"], rebalance_freq=config["rebal_freq"],
        use_hysteresis=config.get("use_hysteresis", False),
        hysteresis_kwargs=config.get("hysteresis_kwargs", None)
    )
    
    test_combined = np.tensordot(score_3d[test_start:], fw, axes=(2, 0))
    test_combined = np.nan_to_num(test_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    test_score = ga_fast_evaluate(
        test_combined, fwd_rets, test_start, n_total, data_mask,
        config["l1_coef"], config["turnover_penalty"],
        top_n=config["top_n"], rebalance_freq=config["rebal_freq"],
        use_hysteresis=config.get("use_hysteresis", False),
        hysteresis_kwargs=config.get("hysteresis_kwargs", None)
    )
    
    logger.info(f"  [{config['name']}] 验证集GA分数={val_score:.4f} 测试集GA分数={test_score:.4f}")
    
    return val_score, test_score

def run_backtest(score_3d, fwd_rets, data_mask, tickers, columns, dates_arr, config, factor_weights):
    logger.info(f"全样本回测: {config['name']}")
    
    n_total = score_3d.shape[0]
    n_symbols = score_3d.shape[1]
    n_factors = score_3d.shape[2]
    
    fw = np.zeros(n_factors, dtype=np.float32)
    if factor_weights is not None:
        fw[:] = np.array(factor_weights, dtype=np.float32)
    
    combined = np.tensordot(score_3d, fw, axes=(2, 0))
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    prev_w = np.zeros(n_symbols, dtype=np.float32)
    hold_since = np.full(n_symbols, -1, dtype=np.int32)
    ret_history_len = 0
    
    equity = np.ones(n_total, dtype=np.float64)
    daily_returns = np.zeros(n_total, dtype=np.float64)
    trade_log = []
    n_trades = 0
    total_tx_cost = 0.0
    
    use_hysteresis = config.get("use_hysteresis", False)
    hysteresis_kwargs = config.get("hysteresis_kwargs", None)
    
    for i in range(1, n_total):
        t = i
        rebalance = (i % config["rebal_freq"] == 0)
        scores_t = combined[t]
        day_data_ok = data_mask[t]
        
        if rebalance:
            if use_hysteresis:
                target = compute_rp_weights(
                    scores_t, fwd_rets[:t+1], ret_history_len,
                    prev_w, hold_since,
                    top_n=config["top_n"], min_hold_days=5
                )
                hk = hysteresis_kwargs or {}
                new_w = make_hysteresis_adjustment(
                    target, prev_w, day_data_ok,
                    large_pos_threshold=hk.get("large_pos_threshold", 0.10),
                    min_adjust_delta=hk.get("min_adjust_delta", 0.02),
                    keep_ratio=hk.get("keep_ratio", 0.70)
                )
            else:
                clean_scores = np.nan_to_num(scores_t, nan=-1e10, neginf=-1e10)
                new_w = compute_rp_weights(
                    clean_scores, fwd_rets[:t+1], ret_history_len,
                    prev_w, hold_since,
                    top_n=config["top_n"], min_hold_days=5
                )
            
            turnover = float(np.sum(np.abs(new_w - prev_w)))
            tx_cost = 0.5 * turnover * TX_COST_RATE
            total_tx_cost += tx_cost
            
            if turnover > 0.01:
                n_trades += 1
                trade_log.append({
                    "date": str(dates_arr[t])[:10],
                    "turnover": round(float(turnover), 4),
                    "tx_cost": round(float(tx_cost), 6),
                    "n_positions": int(np.sum(new_w > 0))
                })
            
            prev_w = new_w
            for sym in range(n_symbols):
                if new_w[sym] > 0 and hold_since[sym] < 0:
                    hold_since[sym] = ret_history_len + 1
        else:
            mask_ok = day_data_ok & (prev_w > 0)
            if np.any(mask_ok):
                p = prev_w[mask_ok].copy()
                p_sum = float(np.sum(p))
                if p_sum > 0:
                    p = p / p_sum
                prev_w = np.zeros(n_symbols, dtype=np.float32)
                prev_w[mask_ok] = p
        
        ret = float(np.dot(prev_w, fwd_rets[t]))
        if np.isnan(ret) or np.isinf(ret):
            ret = 0.0
        ret -= tx_cost if rebalance and turnover > 0.01 else 0.0
        tx_cost = 0.0
        
        daily_returns[i] = ret
        equity[i] = equity[i-1] * (1.0 + ret)
        ret_history_len += 1
    
    total_ret = float(equity[-1] / equity[0] - 1.0)
    n_years = n_total / 252
    annual_ret = (float(equity[-1] / equity[0])) ** (1.0 / max(n_years, 0.5)) - 1.0
    
    log_ret = np.log(equity[1:] / equity[:-1])
    sharpe = float(np.mean(log_ret) / max(np.std(log_ret), 1e-10) * np.sqrt(252))
    
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = float(np.min(drawdown))
    
    mdd_idx = int(np.argmin(drawdown))
    mdd_start = int(np.argmax(cummax[:mdd_idx+1] == equity[:mdd_idx+1]))
    mdd_recover = int(np.where(equity[mdd_idx:] >= cummax[mdd_idx])[0][0]) + mdd_idx if np.any(equity[mdd_idx:] >= cummax[mdd_idx]) else n_total
    
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0
    
    win_days = int(np.sum(daily_returns > 0))
    loss_days = int(np.sum(daily_returns < 0))
    win_rate = win_days / max(win_days + loss_days, 1)
    
    logger.info(f"  年化={annual_ret*100:.2f}% Sharpe={sharpe:.3f} 最大回撤={max_dd*100:.2f}% "
                f"Calmar={calmar:.3f} 胜率={win_rate*100:.1f}% 换手次数={n_trades}")
    
    return {
        "config_name": config["name"],
        "config": config,
        "total_return": total_ret,
        "annual_return": annual_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "total_tx_cost": total_tx_cost,
        "mdd_start": int(mdd_start),
        "mdd_end": int(mdd_idx),
        "mdd_recover": int(mdd_recover),
        "equity_curve": equity.tolist(),
        "daily_returns": daily_returns.tolist(),
        "trade_log": trade_log[-50:],
    }

def generate_report(best_config_name, results, v1_results=None):
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zequant V2 回测报告</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          color: white; padding: 40px 60px; }
.header h1 { font-size: 28px; margin-bottom: 8px; }
.header .subtitle { color: #a0aec0; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 30px; }
.section { background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.section h2 { font-size: 18px; color: #2d3748; margin-bottom: 16px;
              padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid #e2e8f0; }
th { background: #f7fafc; font-weight: 600; color: #4a5568; font-size: 12px; }
td:first-child, th:first-child { text-align: left; }
tr:hover { background: #f7fafc; }
.best { background: #c6f6d5 !important; font-weight: 600; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 11px; font-weight: 600; }
.badge-hys { background: #bee3f8; color: #2b6cb0; }
.badge-raw { background: #e2e8f0; color: #4a5568; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
               gap: 16px; margin-bottom: 20px; }
.metric-card { background: #f7fafc; border-radius: 8px; padding: 16px; text-align: center; }
.metric-card .value { font-size: 24px; font-weight: 700; color: #2d3748; }
.metric-card .label { font-size: 12px; color: #718096; margin-top: 4px; }
.metric-card .positive { color: #38a169; }
.metric-card .negative { color: #e53e3e; }
.summary-table td { vertical-align: middle; }
</style>
</head>
<body>
<div class="header">
<h1>Zequant 量化策略 V2 回测报告</h1>
<div class="subtitle">基于V1最佳因子权重继续优化 · 换手惩罚调优 + 迟滞仓位分配器</div>
</div>
<div class="container">
"""
    
    best_result = None
    for r in results:
        if r["config_name"] == best_config_name:
            best_result = r
            break
    
    if best_result:
        html += '<div class="section"><h2>最佳配置概览</h2>'
        bc = best_result["config"]
        alloc_label = "迟滞分配器" if bc.get("use_hysteresis") else "原始风险平价"
        hk = bc.get("hysteresis_kwargs", {})
        hys_detail = ""
        if hk:
            hys_detail = (f"大仓阈值={hk.get('large_pos_threshold', 'N/A')*100:.0f}%, "
                          f"最小调整={hk.get('min_adjust_delta', 'N/A')*100:.0f}%, "
                          f"保留比例={hk.get('keep_ratio', 'N/A')*100:.0f}%")
        
        html += '<div class="metric-grid">'
        metrics = [
            ("年化收益率", f"{best_result['annual_return']*100:.2f}%", "positive"),
            ("Sharpe比率", f"{best_result['sharpe']:.3f}", "positive"),
            ("最大回撤", f"{best_result['max_drawdown']*100:.2f}%", "negative"),
            ("Calmar", f"{best_result['calmar']:.3f}", "positive"),
            ("胜率", f"{best_result['win_rate']*100:.1f}%", "positive"),
            ("换手次数", f"{best_result['n_trades']}", ""),
        ]
        for label, value, cls in metrics:
            html += f'<div class="metric-card"><div class="value {cls}">{value}</div><div class="label">{label}</div></div>'
        html += '</div>'
        
        html += f'<p style="font-size:13px;color:#4a5568;"><strong>最佳配置:</strong> {best_config_name} | '
        html += f'换手惩罚={bc.get("turnover_penalty", "N/A")} | {alloc_label} | '
        html += f'L1={bc.get("l1_coef", "N/A")} | 调仓频率={bc.get("rebal_freq", "N/A")}天'
        if hys_detail:
            html += f'<br><strong>迟滞参数:</strong> {hys_detail}'
        html += '</p></div>'
    
    html += '<div class="section"><h2>全部实验结果排名</h2>'
    sorted_results = sorted(results, key=lambda x: x["annual_return"], reverse=True)
    html += '<table><tr><th>排名</th><th>配置</th><th>分配器</th><th>惩罚</th><th>年化%</th><th>Sharpe</th><th>回撤%</th><th>Calmar</th><th>胜率%</th><th>换手</th></tr>'
    for rank, r in enumerate(sorted_results, 1):
        cls = "best" if r["config_name"] == best_config_name else ""
        cfg = r["config"]
        is_hys = cfg.get("use_hysteresis", False)
        badge = '<span class="badge badge-hys">迟滞</span>' if is_hys else '<span class="badge badge-raw">原始</span>'
        hys_params = ""
        if is_hys:
            hk = cfg.get("hysteresis_kwargs", {})
            hys_params = f"<br><small>大仓{hk.get('large_pos_threshold', 0)*100:.0f}%/调{hk.get('min_adjust_delta', 0)*100:.0f}%/保{hk.get('keep_ratio', 0)*100:.0f}%</small>"
        html += f'<tr class="{cls}"><td>{rank}</td><td>{r["config_name"]}{hys_params}</td><td>{badge}</td>'
        html += f'<td>{cfg.get("turnover_penalty", "N/A")}</td>'
        html += f'<td>{r["annual_return"]*100:.2f}%</td><td>{r["sharpe"]:.3f}</td>'
        html += f'<td>{r["max_drawdown"]*100:.2f}%</td><td>{r["calmar"]:.3f}</td>'
        html += f'<td>{r["win_rate"]*100:.1f}%</td><td>{r["n_trades"]}</td></tr>'
    html += '</table></div>'
    
    html += '<div class="section"><h2>惩罚系数对比（原始风险平价）</h2>'
    raw_results = [r for r in sorted_results if not r["config"].get("use_hysteresis", False)]
    html += '<table><tr><th>惩罚值</th><th>年化%</th><th>Sharpe</th><th>回撤%</th><th>换手</th></tr>'
    for r in raw_results:
        html += f'<tr><td>{r["config"].get("turnover_penalty", "N/A")}</td>'
        html += f'<td>{r["annual_return"]*100:.2f}%</td><td>{r["sharpe"]:.3f}</td>'
        html += f'<td>{r["max_drawdown"]*100:.2f}%</td><td>{r["n_trades"]}</td></tr>'
    html += '</table></div>'
    
    hys_results = [r for r in sorted_results if r["config"].get("use_hysteresis", False)]
    if hys_results:
        html += '<div class="section"><h2>迟滞分配器结果（按迟滞类型分组）</h2>'
        hys_types = set(r["config"].get("name", "").rsplit("_", 1)[-1] if "_" in str(r.get("config_name", "")) else "" for r in hys_results)
        for htype in sorted(set(
            str(r["config_name"]).split("_")[-1] if "_" in str(r["config_name"]) else ""
            for r in hys_results
        )):
            if not htype:
                continue
            group = [r for r in hys_results if str(r["config_name"]).endswith(htype)]
            if not group:
                continue
            html += f'<h3 style="font-size:14px;margin:12px 0 8px;">{htype}</h3>'
            html += '<table><tr><th>惩罚值</th><th>年化%</th><th>Sharpe</th><th>回撤%</th></tr>'
            for r in group:
                html += f'<tr><td>{r["config"].get("turnover_penalty", "N/A")}</td>'
                html += f'<td>{r["annual_return"]*100:.2f}%</td><td>{r["sharpe"]:.3f}</td>'
                html += f'<td>{r["max_drawdown"]*100:.2f}%</td></tr>'
            html += '</table>'
        html += '</div>'
    
    if v1_results:
        html += '<div class="section"><h2>与V1最佳对比</h2>'
        html += '<table><tr><th>版本</th><th>配置</th><th>年化%</th><th>Sharpe</th><th>回撤%</th></tr>'
        html += f'<tr><td>V1最佳</td><td>周频_L1中_80代</td><td>45.46%</td><td>1.187</td><td>49.26%</td></tr>'
        html += f'<tr class="best"><td>V2最佳</td><td>{best_config_name}</td>'
        if best_result:
            html += f'<td>{best_result["annual_return"]*100:.2f}%</td><td>{best_result["sharpe"]:.3f}</td><td>{best_result["max_drawdown"]*100:.2f}%</td>'
        html += '</tr></table></div>'
    
    html += """
<div class="section"><h2>注意事项</h2>
<ul style="font-size:13px;color:#4a5568;padding-left:20px;">
<li>交易成本: 佣金万3 + 印花税千0.5 + 过户费万0.1 + 滑点万5 = 0.12%</li>
<li>回测时间: 2018-01-01 至 2026-04-30</li>
<li>股票池: 全A股（剔除ST、停牌）</li>
<li>GA优化: 种群30，代数60，早停20代无改善</li>
<li>风险平价: 20日波动率倒数加权，40只股票</li>
<li>调仓频率: 每3个交易日</li>
<li>报告生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</li>
</ul>
</div>
</div>
</body>
</html>"""
    return html

def main():
    logger.info("=" * 60)
    logger.info(f"V2 Pipeline 实验设计: {NUM_EXPERIMENTS}组 (6惩罚值 × 3分配器)")
    logger.info("=" * 60)
    
    score_3d, fwd_rets, tickers, columns, dates_arr, data_mask, factor_names, db = load_data()
    
    n_total = score_3d.shape[0]
    train_end = int(n_total * 0.75)
    train_idx = slice(0, train_end)
    
    checkpoint = load_checkpoint()
    
    if not checkpoint["ga_results"]:
        checkpoint["ga_results"] = []
    
    completed_names = {r["name"] for r in checkpoint["ga_results"]}
    logger.info(f"已完成: {len(completed_names)} 组")
    
    for i, config in enumerate(EXPERIMENTS):
        name = config["name"]
        if name in completed_names:
            logger.info(f"[{i+1}/{NUM_EXPERIMENTS}] {name} 已存在，跳过")
            continue
        
        logger.info(f"[{i+1}/{NUM_EXPERIMENTS}] 开始GA: {name}")
        gc.collect()
        
        result = run_single_ga(
            score_3d, fwd_rets, data_mask, train_idx, None, config
        )
        
        checkpoint["ga_results"].append(result)
        
        if checkpoint["best_overall"] is None or result["best_fitness"] > checkpoint["best_overall"]["best_fitness"]:
            checkpoint["best_overall"] = {
                "name": result["name"],
                "best_fitness": result["best_fitness"],
                "config": config,
                "best_weights": result["best_weights"],
            }
        
        save_checkpoint(checkpoint)
        
        temp_ga = {"experiments": checkpoint["ga_results"], "best_overall": checkpoint["best_overall"]}
        with open(GA_RESULTS_FILE, 'w') as f:
            json.dump(temp_ga, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"[{i+1}/{NUM_EXPERIMENTS}] {name} 完成, best_fitness={result['best_fitness']:.4f}")
    
    logger.info("=" * 60)
    logger.info("全部GA完成，开始回测...")
    logger.info("=" * 60)
    
    ga_results = checkpoint["ga_results"]
    best_overall = checkpoint["best_overall"]
    
    ga_results_sorted = sorted(ga_results, key=lambda x: x["best_fitness"], reverse=True)
    logger.info("GA排名:")
    for rank, r in enumerate(ga_results_sorted, 1):
        logger.info(f"  #{rank}: {r['name']} fitness={r['best_fitness']:.4f}")
    
    logger.info(f"最佳配置: {best_overall['name']} fitness={best_overall['best_fitness']:.4f}")
    
    best_config_json = {
        "version": "v2",
        "base_version": "v1",
        "base_best_config": "周频_L1中_80代",
        "best_experiment": best_overall["name"],
        "best_fitness": best_overall["best_fitness"],
        "config": best_overall["config"],
        "best_weights": best_overall["best_weights"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(BEST_CONFIG_FILE, 'w') as f:
        json.dump(best_config_json, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    logger.info(f"最佳配置已保存: {BEST_CONFIG_FILE}")
    
    full_results = []
    for gr in ga_results:
        logger.info(f"回测: {gr['name']}")
        cfg = None
        for exp in EXPERIMENTS:
            if exp["name"] == gr["name"]:
                cfg = exp
                break
        if cfg is None:
            cfg = gr.get("config", {})
        
        bt_result = run_backtest(
            score_3d, fwd_rets, data_mask, tickers, columns, dates_arr,
            cfg, gr["best_weights"]
        )
        full_results.append(bt_result)
    
    full_results_sorted = sorted(full_results, key=lambda x: x["annual_return"], reverse=True)
    
    bt_output = {
        "version": "v2",
        "results": full_results_sorted,
        "best_config": best_config_json,
    }
    with open(BACKTEST_RESULTS_FILE, 'w') as f:
        json.dump(bt_output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    best_bt = None
    for r in full_results:
        if r["config_name"] == best_overall["name"]:
            best_bt = r
            break
    if best_bt and best_bt.get("equity_curve"):
        with open(EQUITY_CURVE_FILE, 'w') as f:
            json.dump({"config_name": best_overall["name"], "equity_curve": best_bt["equity_curve"]}, f)
    
    v1_ref_file = os.path.join(OUT_DIR, "v1_reference", "backtest_results.json")
    v1_results = None
    if os.path.exists(v1_ref_file):
        try:
            with open(v1_ref_file) as f:
                v1_data = json.load(f)
            v1_results = v1_data.get("results", v1_data)
        except:
            pass
    
    html_report = generate_report(best_overall["name"], full_results_sorted, v1_results)
    with open(BACKTEST_REPORT_FILE, 'w') as f:
        f.write(html_report)
    logger.info(f"回测报告已生成: {BACKTEST_REPORT_FILE}")
    
    logger.info("=" * 60)
    logger.info("V2 Pipeline 全部完成")
    logger.info(f"最佳配置: {best_overall['name']}")
    logger.info(f"最佳GA分数: {best_overall['best_fitness']:.4f}")
    if best_bt:
        logger.info(f"年化收益: {best_bt['annual_return']*100:.2f}%")
        logger.info(f"Sharpe: {best_bt['sharpe']:.3f}")
        logger.info(f"最大回撤: {best_bt['max_drawdown']*100:.2f}%")
    logger.info(f"报告: {BACKTEST_REPORT_FILE}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
