"""Parameter sensitivity scan: stop_loss x crash_threshold x recovery_days.

40 combinations, each runs EmergencyPipeline under the same conditions as mf_d10_rp.
Results ranked by 5-dimension composite score, top combination saved for strategy creation.
"""
from __future__ import annotations
import sys, os, json, logging, itertools, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
logging.basicConfig(level=logging.WARNING)  # 减少日志输出
logging.getLogger('emergency_scan').setLevel(logging.INFO)
logger = logging.getLogger('emergency_scan')

os.environ['NUMBA_DISABLE_JIT'] = '1'

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_experiment(name, stop_loss, crash_threshold, crash_recovery_days,
                   factors, use_universe=True, tx_cost=0.002):
    """Run a single EmergencyPipeline experiment and return metrics."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'emergency_experiment',
        os.path.join(os.path.dirname(__file__), 'emergency_experiment.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    EmergencyPipeline = mod.EmergencyPipeline

    p = EmergencyPipeline(
        name=name, rebal_freq=10, top_n=20, min_hold_days=5,
        positioner_type='rp', factor_names=factors,
        use_universe_filter=use_universe, tx_cost=tx_cost,
        stop_loss=stop_loss if stop_loss is not None else 999.0,
        crash_threshold=crash_threshold if crash_threshold is not None else 999.0,
        crash_reduce_to=0.50,
        crash_recovery_days=crash_recovery_days,
    )
    result = p.run(start='2019-01-02', end='2026-04-30')

    WINDOWS = [
        ('全区间', '2019-01-02', '2026-04-30'),
        ('2022熊市', '2022-01-04', '2022-12-30'),
        ('修复牛OOS', '2024-07-01', '2026-04-30'),
    ]
    windows = p.window_analysis(WINDOWS)

    return {
        'annual_return': result.annual_return,
        'sharpe': result.sharpe,
        'max_drawdown': result.max_drawdown,
        'calmar': result.calmar,
        'win_rate': result.win_rate,
        'recovery_days': result.recovery_days,
        'n_trades': result.n_trades,
        'windows': [{
            'name': w.window,
            'annual_return': w.annual_return,
            'sharpe': w.sharpe,
            'max_drawdown': w.max_drawdown,
        } for w in windows if w.n_days > 0],
    }


def composite_score(m):
    """5-dimension weighted score (same system as INDEX.md)."""
    r = m['annual_return']
    s = m['sharpe']
    dd = abs(m['max_drawdown'])
    rec = m['recovery_days']
    bear = m['windows'][1]['annual_return'] if len(m['windows']) > 1 else 0

    def s_r(v): return min(100, max(0, v / 0.50 * 100))
    def s_s(v): return min(100, max(0, v / 2.0 * 100))
    def s_b(v):
        if v >= 0: return 100
        if v >= -0.10: return 60 + (v + 0.10) / 0.10 * 40
        if v >= -0.20: return 20 + (v + 0.20) / 0.10 * 40
        return max(0, 20 + (v + 0.20) / 0.20 * 20)
    def s_d(v):
        if v <= 0.05: return 100
        if v <= 0.20: return 100 - (v - 0.05) / 0.15 * 40
        if v <= 0.40: return 60 - (v - 0.20) / 0.20 * 40
        return max(0, 20 - (v - 0.40) / 0.20 * 20)
    def s_rec(v):
        if v < 20: return 100
        if v < 60: return 80 - (v - 20) / 40 * 20
        if v < 180: return 60 - (v - 60) / 120 * 30
        if v < 365: return 30 - (v - 180) / 185 * 30
        return max(0, 15 - (v - 365) / 135 * 15)

    total = (s_r(r) * 0.20 + s_s(s) * 0.20 + s_b(bear) * 0.25 +
             s_d(dd) * 0.20 + s_rec(rec) * 0.15)
    return round(total, 1)


def main():
    # Parameter grid
    stop_losses = [None, 0.10, 0.15, 0.20]
    crash_thresholds = [None, 0.04, 0.05, 0.06]
    recovery_days_list = [3, 5, 10]

    # Build combinations
    combos = []
    for sl in stop_losses:
        for ct in crash_thresholds:
            for rd in recovery_days_list:
                # Skip invalid combos
                if ct is None and sl is None and rd != recovery_days_list[0]:
                    continue
                if sl is None and ct is None:
                    # baseline - only run once
                    if rd == recovery_days_list[0]:
                        combos.append((sl, ct, rd))
                    continue
                combos.append((sl, ct, rd))

    logger.info(f"Total combinations: {len(combos)}")

    # Load factors once
    from core.database import Database
    from core.strategies.pipeline import DEFAULT_FACTORS
    db = Database()
    all_factors = db.list_factor_columns()
    factors = [f for f in DEFAULT_FACTORS if f in all_factors]
    db.close()
    logger.info(f"Using {len(factors)} factors")

    # Run all experiments
    results = []
    for i, (sl, ct, rd) in enumerate(combos):
        sl_label = f"sl{sl}" if sl is not None else "slN"
        ct_label = f"ct{ct}" if ct is not None else "ctN"
        name = f"mfd10_{sl_label}_{ct_label}_r{rd}"

        logger.info(f"[{i+1}/{len(combos)}] {name} (sl={sl}, ct={ct}, rd={rd})")

        t0 = time.time()
        try:
            m = run_experiment(name, sl, ct, rd, factors)
            elapsed = time.time() - t0
            score = composite_score(m)

            results.append({
                'name': name,
                'stop_loss': sl,
                'crash_threshold': ct,
                'recovery_days': rd,
                'composite_score': score,
                'annual_return': round(m['annual_return'], 4),
                'sharpe': round(m['sharpe'], 4),
                'max_drawdown': round(m['max_drawdown'], 4),
                'calmar': round(m['calmar'], 4),
                'win_rate': round(m['win_rate'], 4),
                'recovery_days_actual': m['recovery_days'],
                'n_trades': m['n_trades'],
                'windows': m['windows'],
                'elapsed': round(elapsed, 1),
            })
            logger.info(f"  Score={score}, AnnRet={m['annual_return']*100:.2f}%, "
                        f"Sharpe={m['sharpe']:.3f}, MDD={m['max_drawdown']*100:.2f}%")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({
                'name': name, 'stop_loss': sl, 'crash_threshold': ct,
                'recovery_days': rd, 'composite_score': 0,
                'error': str(e),
            })
        elapsed = time.time() - t0

    # Sort by composite score
    valid = [r for r in results if r.get('composite_score', 0) > 0]
    valid.sort(key=lambda x: x['composite_score'], reverse=True)

    # Save full results
    fp = os.path.join(RESULTS_DIR, 'param_scan_results.json')
    with open(fp, 'w') as f:
        json.dump({
            'n_total': len(results),
            'n_valid': len(valid),
            'results': results,
            'ranking': [r['name'] for r in valid],
        }, f, indent=2)
    logger.info(f"Results saved to {fp}")

    # Print Top 15
    print()
    print("=" * 140)
    print("  mf_d10_rp 紧急参数扫描 — Top 15 (按综合评分)")
    print("=" * 140)
    print(f"  {'Rank':<5} {'Name':<32} {'Score':<6} {'Ann%':<7} {'Sharpe':<7} "
          f"{'MDD%':<7} {'Calmar':<7} {'熊市%':<7} {'Rec':<5}  Config")
    print("  " + "-" * 130)
    for rank, r in enumerate(valid[:15], 1):
        bw = ''
        if r.get('windows') and len(r['windows']) > 1:
            bw = f"{r['windows'][1]['annual_return']*100:.1f}%"
        sl = f"sl={r['stop_loss']}" if r['stop_loss'] else "no_sl"
        ct = f"ct={r['crash_threshold']}" if r['crash_threshold'] else "no_ct"
        rd = f"rd={r['recovery_days']}"
        print(f"  {rank:<5} {r['name']:<32} {r['composite_score']:<6} "
              f"{r['annual_return']*100:<6.2f}% {r['sharpe']:<7.3f} "
              f"{abs(r['max_drawdown'])*100:<6.2f}% {r['calmar']:<7.3f} "
              f"{bw:<7} {r['recovery_days_actual']:<5}  {sl}/{ct}/{rd}")
    print("  " + "-" * 130)
    print()

    # Top 1 detail
    best = valid[0]
    print(f"🏆 最优参数: {best['name']}")
    print(f"   stop_loss={best['stop_loss']}, crash_threshold={best['crash_threshold']}, "
          f"recovery_days={best['recovery_days']}")
    print(f"   综合分={best['composite_score']}, 年化={best['annual_return']*100:.2f}%, "
          f"Sharpe={best['sharpe']:.3f}, 回撤={abs(best['max_drawdown'])*100:.2f}%")
    if best.get('windows'):
        for w in best['windows']:
            print(f"   {w['name']}: 年化={w['annual_return']*100:.2f}% "
                  f"Sharpe={w['sharpe']:.3f}")

    # Compare with baseline (no emergency)
    baseline = [r for r in valid if r['stop_loss'] is None and r['crash_threshold'] is None]
    if baseline:
        b = baseline[0]
        print()
        print("与基准对比:")
        print(f"   基准:     综合分={b['composite_score']}, 年化={b['annual_return']*100:.2f}%, "
              f"Sharpe={b['sharpe']:.3f}, 回撤={abs(b['max_drawdown'])*100:.2f}%")
        print(f"   最优:     综合分={best['composite_score']}, 年化={best['annual_return']*100:.2f}%, "
              f"Sharpe={best['sharpe']:.3f}, 回撤={abs(best['max_drawdown'])*100:.2f}%")
        print(f"   改善:     综合分+{best['composite_score']-b['composite_score']:.1f}, "
              f"Sharpe+{best['sharpe']-b['sharpe']:.3f}, "
              f"回撤{abs(best['max_drawdown'])-abs(b['max_drawdown']):+.2%}")

    print()
    print(f"完整结果: {fp}")


if __name__ == '__main__':
    main()
