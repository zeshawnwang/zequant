"""2026-05-18 实验：mf_d10_rp 添加紧急事件处理并对比回测。

比较两个版本：
  A. mf_d10_rp (原版) — 纯D10调仓，无任何止损/风控
  B. mf_d10_rp + 紧急处理 — 在A基础上增加个股止损和大盘熔断

紧急规则设计（均为收盘后判定，次日开盘执行）：
  1. 个股止损 (stop_loss=-15%)：
     - 每只股票从买入日起记录累计收益
     - 当日累计收益 ≤ -15% 时，次日将其仓位归零
     - 重新分配给其他未触发的持仓
  2. 大盘熔断 (market_crash=-5%)：
     - 计算每日市场等权中位数收益作为"大盘"指标
     - 当日大盘跌幅 ≥ 5% 时，次日总仓位降至 50%
     - 剩余50%做现金，5个交易日后恢复满仓

输出：对比两个版本的三大区间表现 + 逐年对比
"""
from __future__ import annotations
import sys, os, json, logging, numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from core.strategies.pipeline import StrategyPipeline, BacktestMetrics
from core.database import Database

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('emergency_experiment')

# ──────────────────────────────────────────────────
# 策略 B：带紧急处理的 Pipeline 子类
# ──────────────────────────────────────────────────
class EmergencyPipeline(StrategyPipeline):
    """在 StrategyPipeline 基础上增加每日紧急事件处理。"""

    def __init__(self, stop_loss: float = 0.15, crash_threshold: float = 0.05,
                 crash_reduce_to: float = 0.50, crash_recovery_days: int = 5, **kwargs):
        """
        Args:
            stop_loss: 个股止损阈值（累计收益 <= -stop_loss 时卖出）
            crash_threshold: 大盘熔断阈值（日跌幅 >= crash_threshold 时触发）
            crash_reduce_to: 熔断后目标仓位比例
            crash_recovery_days: 熔断后多少天恢复满仓
        """
        super().__init__(**kwargs)
        self.stop_loss = stop_loss
        self.crash_threshold = crash_threshold
        self.crash_reduce_to = crash_reduce_to
        self.crash_recovery_days = crash_recovery_days

    def _backtest_rp(
        self, sig: np.ndarray, fwd: np.ndarray, dm: np.ndarray, nd: int, ns: int
    ) -> Tuple[np.ndarray, int]:
        from core.positioners import RPPortfolioWeights
        alloc = RPPortfolioWeights(top_n=self.top_n, min_hold_days=self.min_hold_days)
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0
        dr = np.zeros(nd, dtype=np.float64)
        nt = 0

        # 紧急事件追踪
        pos_val = np.ones(ns, dtype=np.float64)      # 每只股票的持仓净值基准（1.0=买入价）
        crash_timer = 0                                # 熔断恢复倒计时
        crash_active = False                           # 当前是否处于熔断减仓状态

        for i in range(1, nd):
            # ── 0. 更新持仓净值（用前一日实现的收益 fwd[i-1] 更新） ──
            if i > 1:
                for j in range(ns):
                    if pw[j] > 1e-8:
                        pos_val[j] *= (1.0 + float(fwd[i - 1, j]))

            # ── 1. 个股止损检查 ──
            for j in range(ns):
                if pw[j] > 1e-8 and pos_val[j] < (1.0 - self.stop_loss):
                    logger.debug(f"  止损触发: {self.tks[j] if hasattr(self, 'tks') and j < len(self.tks) else j} "
                                 f"pos_val={pos_val[j]:.4f}")
                    pw[j] = 0.0
                    pos_val[j] = 1.0

            # ── 2. 大盘熔断检查 ──
            # 用上一日全市场中位数收益作为大盘指标
            if i > 1:
                valid_ret = fwd[i - 1][dm[i - 1] & (np.abs(fwd[i - 1]) < 0.15)]
                if len(valid_ret) > 0:
                    market_ret = float(np.median(valid_ret))
                    if market_ret <= -self.crash_threshold and not crash_active:
                        crash_active = True
                        crash_timer = self.crash_recovery_days
                        logger.info(f"  大盘熔断触发: 日{i} 市场收益={market_ret*100:.2f}% "
                                    f"→ 减仓至{self.crash_reduce_to*100:.0f}%")

            # 熔断状态下的仓位缩放
            crash_scale = 1.0
            if crash_active:
                crash_timer -= 1
                crash_scale = self.crash_reduce_to
                if crash_timer <= 0:
                    crash_active = False
                    logger.info(f"  熔断恢复: 日{i} 恢复满仓")

            # ── 3. 调仓 ──
            rebal = (i % self.rebal_freq == 0)
            txc = 0.0
            if rebal:
                masked_sig = sig[i].copy()
                if self.use_universe_filter and self.um is not None:
                    abs_i = getattr(self, '_backtest_sidx', 0) + i
                    masked_sig[~self.um[abs_i]] = -1e10

                nw = alloc.allocate(masked_sig, fwd, i, pw, hs, rh)
                # 应用熔断缩放
                nw = nw * crash_scale
                # 如果减仓了，多出的现金等权重分配到未止损的持仓上
                if crash_scale < 1.0:
                    remaining = np.where((nw > 0) & (pos_val >= 1.0 - self.stop_loss))[0]
                    if len(remaining) > 0:
                        # 已经减掉的部分重新分配
                        reduced = np.sum(pw) * (1 - crash_scale)
                        nw[remaining] += reduced / len(remaining)

                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost
                if to > 0.01:
                    nt += 1
                pw = nw
                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0:
                        hs[j] = rh + 1
                        pos_val[j] = 1.0  # 新持仓重置净值
            else:
                # 非调仓日：保留止损过滤后的持仓，重新归一化
                mk = dm[i] & (pw > 0)
                if crash_active:
                    # 熔断期间限制总仓位
                    if np.any(mk):
                        p2 = pw[mk].copy() * crash_scale / float(np.sum(pw[mk] * crash_scale))
                        pw = np.zeros(ns, dtype=np.float32)
                        pw[mk] = p2
                else:
                    if np.any(mk):
                        p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                        pw = np.zeros(ns, dtype=np.float32)
                        pw[mk] = p2

            # ── 4. 计算当日组合收益 ──
            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1

        return dr, nt


# ──────────────────────────────────────────────────
# 对比实验
# ──────────────────────────────────────────────────
def run_comparison():
    db = Database()
    all_factors = db.list_factor_columns()
    from core.strategies.pipeline import DEFAULT_FACTORS
    factors = [f for f in DEFAULT_FACTORS if f in all_factors]
    db.close()

    RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
    WINDOWS = [
        ('全区间', '2019-01-02', '2026-04-30'),
        ('2022熊市', '2022-01-04', '2022-12-30'),
        ('修复牛OOS', '2024-07-01', '2026-04-30'),
    ]

    configs = [
        ('mf_d10_rp', None, '无紧急处理（原版）'),
        ('mf_d10_rp_emergency', {
            'stop_loss': 0.15,
            'crash_threshold': 0.05,
            'crash_reduce_to': 0.50,
            'crash_recovery_days': 5,
        }, '个股止损-15% + 大盘熔断-5%'),
    ]

    experiments = []
    for name, ekwargs, desc in configs:
        logger.info(f"=" * 60)
        logger.info(f"运行: {name} — {desc}")

        kwargs = dict(
            name=name, rebal_freq=10, top_n=20, min_hold_days=5,
            positioner_type='rp', factor_names=factors,
            use_universe_filter=True, tx_cost=0.002,
        )
        if ekwargs:
            p = EmergencyPipeline(**kwargs, **ekwargs)
        else:
            p = StrategyPipeline(**kwargs)

        result = p.run(start='2019-01-02', end='2026-04-30')
        windows = p.window_analysis(WINDOWS)

        data = {
            'strategy': name,
            'description': desc,
            'config': kwargs,
            'emergency_config': ekwargs or {},
            'full_range': {
                'annual_return': round(result.annual_return, 4),
                'sharpe': round(result.sharpe, 4),
                'max_drawdown': round(result.max_drawdown, 4),
                'calmar': round(result.calmar, 4),
                'win_rate': round(result.win_rate, 4),
                'n_trades': result.n_trades,
                'drawdown_duration': result.drawdown_duration,
                'recovery_days': result.recovery_days,
            },
            'windows': [{
                'name': w.window,
                'annual_return': round(w.annual_return, 4),
                'sharpe': round(w.sharpe, 4),
                'max_drawdown': round(w.max_drawdown, 4),
            } for w in windows if w.n_days > 0],
        }
        fp = os.path.join(RESULTS_DIR, f'{name}.json')
        with open(fp, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        experiments.append(data)

        logger.info(f"  {name} 全区间: 年化={result.annual_return*100:.2f}% "
                    f"Sharpe={result.sharpe:.3f} 回撤={result.max_drawdown*100:.2f}% "
                    f"修复天数={result.recovery_days}")
        for w in windows:
            if w.n_days > 0:
                logger.info(f"  {w.window}: 年化={w.annual_return*100:.2f}% "
                            f"Sharpe={w.sharpe:.3f}")

    return experiments


def print_report(experiments):
    print()
    print("=" * 120)
    print("  mf_d10_rp 紧急事件处理对比回测报告")
    print("=" * 120)
    print()
    print(f"{'对比项':<28} {'mf_d10_rp (原版)':<28} {'mf_d10_rp+紧急处理':<28} {'差异':<16}")
    print("-" * 120)

    metrics = ['annual_return', 'sharpe', 'max_drawdown', 'calmar', 'win_rate', 'recovery_days']
    labels = {
        'annual_return': '年化收益率',
        'sharpe': 'Sharpe',
        'max_drawdown': '最大回撤',
        'calmar': 'Calmar',
        'win_rate': '胜率',
        'recovery_days': '修复天数',
    }
    fmt = {
        'annual_return': lambda v: f"{v*100:.2f}%",
        'sharpe': lambda v: f"{v:.3f}",
        'max_drawdown': lambda v: f"{v*100:.2f}%",
        'calmar': lambda v: f"{v:.3f}",
        'win_rate': lambda v: f"{v*100:.1f}%",
        'recovery_days': lambda v: f"{int(v)}",
    }

    o = {m: experiments[0]['full_range'][m] for m in metrics}
    e = {m: experiments[1]['full_range'][m] for m in metrics}

    for m in metrics:
        ov = o[m]
        ev = e[m]
        label = labels[m]
        if m == 'max_drawdown':
            diff_str = f"{abs(ev) - abs(ov):+.2%}" if isinstance(ov, float) else ""
        elif isinstance(ov, float):
            diff_str = f"{ev - ov:+.4f}" if m == 'sharpe' else f"{ev - ov:+.2%}" if m != 'recovery_days' else ""
        else:
            diff_str = f"{ev - ov:+d}"
        print(f"  {label:<26} {fmt[m](ov):<28} {fmt[m](ev):<28} {diff_str:<16}")

    print("-" * 120)
    print()

    # 区间对比
    print(f"{'区间对比':<28} {'mf_d10_rp 年化':<20} {'mf_d10_rp Sharpe':<20} {'+紧急 年化':<20} {'+紧急 Sharpe':<20}")
    print("-" * 120)
    for wi, (wname, _, _) in enumerate([('全区间', '', ''), ('2022熊市', '', ''), ('修复牛OOS', '', '')]):
        o_w = experiments[0]['windows'][wi]
        e_w = experiments[1]['windows'][wi]
        print(f"  {wname:<26} {o_w['annual_return']*100:<19.2f}% {o_w['sharpe']:<19.3f} "
              f"{e_w['annual_return']*100:<19.2f}% {e_w['sharpe']:<19.3f}")

    print("-" * 120)
    print()

    # 综合评分
    print("综合评分（5维加权，同INDEX.md体系）:")
    print("-" * 80)
    for exp in experiments:
        v = exp['full_range']
        # 简化评分
        def score_return(r): return min(100, max(0, r / 0.50 * 100))
        def score_sharpe(r): return min(100, max(0, r / 2.0 * 100))
        def score_bear(r):
            bw = exp['windows'][1] if len(exp['windows']) > 1 else None
            raw = bw['annual_return'] if bw else 0
            return 100 if raw >= 0 else max(0, 60 + raw / 0.10 * 40) if raw >= -0.10 else max(0, 20)
        def score_dd(r):
            raw = abs(r)
            if raw <= 0.05: return 100
            if raw <= 0.20: return 100 - (raw - 0.05) / 0.15 * 40
            if raw <= 0.40: return 60 - (raw - 0.20) / 0.20 * 40
            return max(0, 20 - (raw - 0.40) / 0.20 * 20)
        def score_rec(r):
            if r < 20: return 100
            if r < 60: return 80 - (r - 20) / 40 * 20
            if r < 180: return 60 - (r - 60) / 120 * 30
            if r < 365: return 30 - (r - 180) / 185 * 30
            return max(0, 15 - (r - 365) / 135 * 15)

        total = (score_return(v['annual_return']) * 0.20 +
                 score_sharpe(v['sharpe']) * 0.20 +
                 score_bear(None) * 0.25 +
                 score_dd(v['max_drawdown']) * 0.20 +
                 score_rec(v.get('recovery_days', 999)) * 0.15)
        print(f"  {exp['description']:<26}: {total:<5.1f} 分")

    print()
    print("=" * 120)


if __name__ == '__main__':
    experiments = run_comparison()
    print_report(experiments)
    print("实验结果已保存至:", os.path.dirname(os.path.abspath(__file__)))
