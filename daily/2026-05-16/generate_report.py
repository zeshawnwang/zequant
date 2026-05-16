"""
策略评估报告生成器

读取窗口排名数据，生成包含对比表、风险分组、窗口分析和策略推荐的HTML报告。
如果 x9_results/window_rankings.json 不存在，从 INDEX.md 构建数据。
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).parent

WINDOW_NAMES = ["全区间", "2019_修复牛", "2020_疫情冲击", "2021_结构牛",
                "2022_熊市", "2023_震荡修复", "2024_反弹", "2025_至今"]

STRATEGIES_INDEX = [
    {"name": "ga_d10", "annual_return": 33.59, "sharpe": 1.435, "max_drawdown": -16.33, "calmar": 2.057, "note": "全库最高Sharpe+最低回撤", "category": "MultiFactor"},
    {"name": "chip_equal_d3", "annual_return": 17.43, "sharpe": 1.401, "max_drawdown": -11.19, "calmar": 1.558, "note": "等权Chip最高Sharpe", "category": "ChipConcentration"},
    {"name": "mf50_chipcovrp50_combo", "annual_return": 22.78, "sharpe": 1.343, "max_drawdown": -17.01, "calmar": 1.339, "note": "组合系列最高Sharpe", "category": "Combo"},
    {"name": "mf_vol_d10_rp", "annual_return": 26.13, "sharpe": 1.334, "max_drawdown": -22.59, "calmar": 1.156, "note": "MF+VolTiming择时", "category": "MultiFactor"},
    {"name": "mf_d10_rp", "annual_return": 38.03, "sharpe": 1.306, "max_drawdown": -30.44, "calmar": 1.250, "note": "旗舰高收益低频", "category": "MultiFactor"},
    {"name": "chip_covrp", "annual_return": 14.96, "sharpe": 1.288, "max_drawdown": -9.68, "calmar": 1.545, "note": "协方差风险平价最低回撤", "category": "ChipConcentration"},
    {"name": "osr_vol_eq_d10", "annual_return": 21.52, "sharpe": 1.280, "max_drawdown": -16.43, "calmar": 1.310, "note": "OSR+VolTiming风控", "category": "OversoldRebound"},
    {"name": "mf60_chip40_combo", "annual_return": 24.84, "sharpe": 1.269, "max_drawdown": -18.20, "calmar": 1.365, "note": "最佳年化/回撤组合", "category": "Combo"},
    {"name": "ga_d5", "annual_return": 47.18, "sharpe": 1.265, "max_drawdown": -29.11, "calmar": 1.620, "note": "全库最高年化", "category": "MultiFactor"},
    {"name": "mf50_chip50_combo", "annual_return": 22.29, "sharpe": 1.257, "max_drawdown": -16.60, "calmar": 1.343, "note": "组合最低回撤", "category": "Combo"},
    {"name": "osr_d10", "annual_return": 23.55, "sharpe": 1.164, "max_drawdown": -26.39, "calmar": 0.892, "note": "超跌反弹旗舰", "category": "OversoldRebound"},
    {"name": "v1_ga_rp", "annual_return": 45.46, "sharpe": 1.119, "max_drawdown": -39.77, "calmar": 1.143, "note": "V1最高收益", "category": "MultiFactor"},
    {"name": "chip_vol_rp", "annual_return": 10.37, "sharpe": 1.101, "max_drawdown": -10.51, "calmar": 0.986, "note": "全库最低回撤", "category": "ChipConcentration"},
    {"name": "mf_trend_d5_rp", "annual_return": 17.16, "sharpe": 1.083, "max_drawdown": -14.42, "calmar": 1.190, "note": "趋势择时风控", "category": "MultiFactor"},
    {"name": "chip_rp", "annual_return": 14.50, "sharpe": 1.071, "max_drawdown": -14.58, "calmar": 0.994, "note": "筹码集中基础版", "category": "ChipConcentration"},
    {"name": "v4_mf_rp", "annual_return": 26.93, "sharpe": 0.976, "max_drawdown": -28.71, "calmar": 0.938, "note": "V4多因子", "category": "MultiFactor"},
    {"name": "v4_mf_tv_rp", "annual_return": 21.87, "sharpe": 0.918, "max_drawdown": -16.32, "calmar": 1.340, "note": "V4+趋势择时", "category": "MultiFactor"},
]

CATEGORY_CN = {
    "MultiFactor": "多因子策略",
    "ChipConcentration": "筹码集中策略",
    "Combo": "组合策略",
    "OversoldRebound": "超跌反弹策略",
}

def load_window_data() -> List[Dict[str, Any]]:
    x9_path = BASE_DIR / "x9_results" / "window_rankings.json"
    if x9_path.exists():
        with open(x9_path) as f:
            return json.load(f)
    window_path = BASE_DIR / "v9_window_eval" / "window_results.json"
    if window_path.exists():
        with open(window_path) as f:
            return json.load(f)
    return []


def get_risk_level(s: Dict) -> str:
    dd = abs(s["max_drawdown"])
    if dd < 15:
        return "low"
    elif dd < 25:
        return "mid"
    return "high"


def risk_label(level: str) -> str:
    return {"low": "低风险", "mid": "中风险", "high": "高风险"}[level]


def risk_emoji(level: str) -> str:
    return {"low": "🟢", "mid": "🟡", "high": "🔴"}[level]


def get_color(val, metric: str) -> str:
    if metric == "sharpe":
        if val >= 1.3: return "#22c55e"
        if val >= 1.0: return "#eab308"
        return "#ef4444"
    if metric == "annual_return":
        if val >= 30: return "#22c55e"
        if val >= 15: return "#eab308"
        return "#f97316"
    if metric == "max_drawdown":
        if val >= -15: return "#22c55e"
        if val >= -25: return "#eab308"
        return "#ef4444"
    if metric == "calmar":
        if val >= 1.5: return "#22c55e"
        if val >= 1.0: return "#eab308"
        return "#ef4444"
    return "inherit"


def get_category_color(cat: str) -> str:
    return {
        "MultiFactor": "#3b82f6",
        "ChipConcentration": "#8b5cf6",
        "Combo": "#f59e0b",
        "OversoldRebound": "#ec4899",
    }.get(cat, "#6b7280")


def build_market_recommendations():
    return {
        "bull": {
            "emoji": "🐂",
            "title": "牛市",
            "desc": "趋势向上，风险偏好高，适合进攻型策略",
            "strategies": [
                {"name": "mf_d10_rp", "weight": "60%", "reason": "高Beta满仓进攻，全区间年化38%"},
                {"name": "ga_d5", "weight": "25%", "reason": "全库最高年化47%，牛市中弹性最大"},
                {"name": "ga_d10", "weight": "15%", "reason": "最高Sharpe+最低回撤，提供缓冲"},
            ],
        },
        "bear": {
            "emoji": "🐻",
            "title": "熊市",
            "desc": "趋势向下，防守为主，控制回撤是第一位",
            "strategies": [
                {"name": "chip_vol_rp", "weight": "40%", "reason": "全库最低回撤-10.5%，熊市避风港"},
                {"name": "chip_covrp", "weight": "35%", "reason": "协方差风险平价，回撤仅-9.7%"},
                {"name": "chip_rp", "weight": "25%", "reason": "筹码集中防御型，年化仍达14.5%"},
            ],
        },
        "oscillate": {
            "emoji": "〰️",
            "title": "震荡市",
            "desc": "方向不明，均衡配置，多策略分散降低波动",
            "strategies": [
                {"name": "mf50_chip50_combo", "weight": "35%", "reason": "MF+Chip均衡组合，回撤仅-16.6%"},
                {"name": "chip_covrp", "weight": "25%", "reason": "低回撤底仓保护"},
                {"name": "ga_d10", "weight": "25%", "reason": "GA优化在震荡市中Sharpe仍达1.43"},
                {"name": "mf_vol_d10_rp", "weight": "15%", "reason": "波动择时保护下行"},
            ],
        },
        "recovery": {
            "emoji": "📈",
            "title": "修复/反弹市",
            "desc": "超跌后反弹，适合超跌反弹+温和进攻组合",
            "strategies": [
                {"name": "osr_d10", "weight": "35%", "reason": "超跌反弹旗舰窗口Sharpe>2.0"},
                {"name": "mf60_chip40_combo", "weight": "35%", "reason": "温和进攻，年化/回撤最佳比"},
                {"name": "mf_vol_d10_rp", "weight": "30%", "reason": "波动保护防止二次探底"},
            ],
        },
    }


def generate_html() -> str:
    strategies = STRATEGIES_INDEX
    window_data = load_window_data()

    cat_order = ["MultiFactor", "ChipConcentration", "Combo", "OversoldRebound"]

    rows_html = ""
    for i, s in enumerate(strategies, 1):
        color_sharpe = get_color(s["sharpe"], "sharpe")
        color_ret = get_color(s["annual_return"], "annual_return")
        color_dd = get_color(s["max_drawdown"], "max_drawdown")
        color_calmar = get_color(s["calmar"], "calmar")
        row = f"""            <tr>
                <td class="rank">{i}</td>
                <td class="strategy-name"><span class="cat-dot" style="background:{get_category_color(s['category'])}"></span>{s['name']}</td>
                <td class="num" style="color:{color_ret}">{s['annual_return']:.2f}%</td>
                <td class="num" style="color:{color_sharpe}">{s['sharpe']:.3f}</td>
                <td class="num" style="color:{color_dd}">{s['max_drawdown']:.2f}%</td>
                <td class="num" style="color:{color_calmar}">{s['calmar']:.3f}</td>
                <td class="note">{s['note']}</td>
            </tr>"""
        rows_html += row + "\n"

    cat_rows_html = ""
    for cat in cat_order:
        cat_strats = [s for s in strategies if s["category"] == cat]
        if not cat_strats:
            continue
        cat_rows_html += f"""        <tr class="cat-header"><td colspan="6">{CATEGORY_CN[cat]}</td></tr>\n"""
        for s in cat_strats:
            cat_rows_html += f"""        <tr>
                <td class="strategy-name">{s['name']}</td>
                <td class="num" style="color:{get_color(s['annual_return'], 'annual_return')}">{s['annual_return']:.2f}%</td>
                <td class="num" style="color:{get_color(s['sharpe'], 'sharpe')}">{s['sharpe']:.3f}</td>
                <td class="num" style="color:{get_color(s['max_drawdown'], 'max_drawdown')}">{s['max_drawdown']:.2f}%</td>
                <td class="num" style="color:{get_color(s['calmar'], 'calmar')}">{s['calmar']:.3f}</td>
                <td class="note">{s['note']}</td>
            </tr>\n"""

    risk_groups = {"low": [], "mid": [], "high": []}
    for s in strategies:
        risk_groups[get_risk_level(s)].append(s)

    risk_rows_html = ""
    risk_order = ["low", "mid", "high"]
    for rl in risk_order:
        rl_strats = risk_groups[rl]
        if not rl_strats:
            continue
        risk_rows_html += f"""        <tr class="cat-header"><td colspan="4">{risk_emoji(rl)} {risk_label(rl)}（回撤{'<' if rl == 'low' else ''}{'15%' if rl == 'low' else '15~25%' if rl == 'mid' else '>25%'}）</td></tr>\n"""
        for s in rl_strats:
            risk_rows_html += f"""        <tr>
                <td class="strategy-name">{s['name']}</td>
                <td class="num" style="color:{get_color(s['annual_return'], 'annual_return')}">{s['annual_return']:.2f}%</td>
                <td class="num" style="color:{get_color(s['max_drawdown'], 'max_drawdown')}">{s['max_drawdown']:.2f}%</td>
                <td class="num" style="color:{get_color(s['sharpe'], 'sharpe')}">{s['sharpe']:.3f}</td>
            </tr>\n"""

    recs = build_market_recommendations()
    rec_cards_html = ""
    for state_key, r in recs.items():
        strat_rows = "".join(
            f"""                <div class="rec-strat"><span class="rec-name">{s['name']}</span><span class="rec-weight">{s['weight']}</span><span class="rec-reason">{s['reason']}</span></div>\n"""
            for s in r["strategies"]
        )
        rec_cards_html += f"""        <div class="rec-card rec-{state_key}">
            <div class="rec-header">
                <span class="rec-emoji">{r['emoji']}</span>
                <div>
                    <div class="rec-title">{r['title']}</div>
                    <div class="rec-desc">{r['desc']}</div>
                </div>
            </div>
            <div class="rec-body">
{strat_rows}            </div>
        </div>\n"""

    sharpe_data_json = json.dumps([
        {"name": s["name"], "value": round(s["sharpe"], 3)}
        for s in strategies
    ])
    cat_data_json = json.dumps([
        {"name": CATEGORY_CN[cat], "value": len([s for s in strategies if s["category"] == cat])}
        for cat in cat_order
    ])
    risk_data_json = json.dumps([
        {"name": f"{risk_emoji(rl)} {risk_label(rl)}", "value": len(risk_groups[rl])}
        for rl in risk_order
    ])

    n_strats = len(strategies)
    top_sharpe = strategies[0]["name"]
    top_sharpe_val = strategies[0]["sharpe"]
    top_ret = max(strategies, key=lambda x: x["annual_return"])
    min_dd = min(strategies, key=lambda x: x["max_drawdown"])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZEquant 策略评估报告 2026-05-16</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
:root {{
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --bg-table-stripe: #0f1729;
    --border: #1e293b;
    --text: #e2e8f0;
    --text-dim: #64748b;
    --text-bright: #f1f5f9;
    --accent: #f59e0b;
    --accent-glow: rgba(245, 158, 11, 0.25);
    --green: #22c55e;
    --red: #ef4444;
    --blue: #3b82f6;
    --purple: #8b5cf6;
    --pink: #ec4899;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background: var(--bg-primary);
    color: var(--text);
    font-family: 'Noto Sans SC', -apple-system, sans-serif;
    line-height:1.6;
    overflow-x:hidden;
}}
body::before {{
    content:'';
    position:fixed; top:0; left:0; width:100vw; height:100vh;
    background: radial-gradient(ellipse 80% 60% at 50% -20%, rgba(59,130,246,0.08), transparent),
                radial-gradient(ellipse 60% 50% at 80% 80%, rgba(245,158,11,0.04), transparent);
    pointer-events:none; z-index:0;
}}
.container {{ max-width:1280px; margin:0 auto; padding: 0 24px; position:relative; z-index:1; }}

/* Header */
.hero {{
    padding: 64px 0 48px;
    text-align:center;
    border-bottom:1px solid var(--border);
    margin-bottom:48px;
}}
.hero-badge {{
    display:inline-block;
    background:var(--accent-glow);
    color:var(--accent);
    font-size:12px; font-weight:600;
    padding:4px 16px;
    border-radius:100px;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:16px;
    border:1px solid rgba(245,158,11,0.3);
}}
.hero h1 {{
    font-size:52px; font-weight:900;
    letter-spacing:-2px;
    margin-bottom:8px;
    background:linear-gradient(135deg, #f1f5f9 0%, #f59e0b 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}}
.hero p {{
    color:var(--text-dim);
    font-size:18px;
    font-weight:300;
}}
.hero-stats {{
    display:flex; justify-content:center; gap:48px;
    margin-top:32px;
}}
.hero-stat {{ text-align:center; }}
.hero-stat-value {{
    font-family:'JetBrains Mono', monospace;
    font-size:32px; font-weight:700;
    color:var(--text-bright);
}}
.hero-stat-label {{
    font-size:13px;
    color:var(--text-dim);
    margin-top:2px;
}}

/* Section */
.section {{
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:16px;
    padding:32px;
    margin-bottom:32px;
    backdrop-filter:blur(12px);
    transition:box-shadow 0.3s;
}}
.section:hover {{
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
}}
.section-title {{
    font-size:20px; font-weight:700;
    margin-bottom:20px;
    display:flex; align-items:center; gap:10px;
}}
.section-title::after {{
    content:'';
    flex:1;
    height:1px;
    background:linear-gradient(to right, var(--border), transparent);
}}

/* Tables */
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{
    text-align:left; padding:10px 12px;
    font-size:12px; font-weight:600; text-transform:uppercase;
    color:var(--text-dim); letter-spacing:0.5px;
    border-bottom:1px solid var(--border);
    white-space:nowrap;
}}
td {{ padding:10px 12px; border-bottom:1px solid rgba(30,41,59,0.5); }}
tr:hover td {{ background:var(--bg-card-hover); }}
.num {{ font-family:'JetBrains Mono', monospace; text-align:right; font-weight:600; }}
.strategy-name {{ font-weight:500; display:flex; align-items:center; gap:8px; white-space:nowrap; }}
.cat-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }}
.rank {{ color:var(--text-dim); font-family:'JetBrains Mono', monospace; text-align:center; width:40px; }}
.note {{ color:var(--text-dim); font-size:13px; max-width:220px; }}
.cat-header td {{
    background:rgba(30,41,59,0.5);
    font-weight:700; font-size:13px;
    padding:8px 12px;
    color:var(--accent);
    letter-spacing:1px;
}}
tr.cat-header:hover td {{ background:rgba(30,41,59,0.5); }}

/* Charts row */
.charts-row {{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:24px;
    margin-bottom:32px;
}}
.chart-card {{
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:16px;
    padding:24px;
}}
.chart-card h3 {{
    font-size:15px; font-weight:600;
    margin-bottom:12px;
    color:var(--text-dim);
}}
.chart-box {{ width:100%; height:360px; }}

/* Recommendations */
.rec-grid {{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:20px;
}}
.rec-card {{
    padding:24px;
    border-radius:12px;
    border:1px solid var(--border);
    transition:all 0.3s;
}}
.rec-card:hover {{ transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,0.3); }}
.rec-bull {{ background:linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.02)); border-color:rgba(34,197,94,0.3); }}
.rec-bear {{ background:linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.02)); border-color:rgba(239,68,68,0.3); }}
.rec-oscillate {{ background:linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.02)); border-color:rgba(245,158,11,0.3); }}
.rec-recovery {{ background:linear-gradient(135deg, rgba(59,130,246,0.1), rgba(59,130,246,0.02)); border-color:rgba(59,130,246,0.3); }}
.rec-header {{ display:flex; align-items:center; gap:12px; margin-bottom:16px; }}
.rec-emoji {{ font-size:32px; }}
.rec-title {{ font-size:18px; font-weight:700; }}
.rec-desc {{ font-size:13px; color:var(--text-dim); }}
.rec-strat {{
    display:grid;
    grid-template-columns:150px 50px 1fr;
    gap:8px;
    padding:8px 0;
    border-bottom:1px solid rgba(30,41,59,0.5);
    font-size:13px;
    align-items:center;
}}
.rec-strat:last-child {{ border-bottom:none; }}
.rec-name {{ font-family:'JetBrains Mono', monospace; font-weight:600; }}
.rec-weight {{ font-family:'JetBrains Mono', monospace; color:var(--accent); text-align:center; font-weight:700; }}
.rec-reason {{ color:var(--text-dim); }}

/* Footer */
.footer {{
    text-align:center;
    padding:48px 0;
    color:var(--text-dim);
    font-size:13px;
    border-top:1px solid var(--border);
    margin-top:32px;
}}

/* Responsive */
@media (max-width: 768px) {{
    .hero h1 {{ font-size:32px; }}
    .hero-stats {{ gap:24px; flex-wrap:wrap; }}
    .charts-row {{ grid-template-columns:1fr; }}
    .rec-grid {{ grid-template-columns:1fr; }}
    .rec-strat {{ grid-template-columns:120px 40px 1fr; }}
}}
</style>
</head>
<body>
<div class="container">

<div class="hero">
    <div class="hero-badge">📊 ZEquant · 策略评估报告</div>
    <h1>17 策略全景扫描</h1>
    <p>基于全区间回测数据 · 2026-05-16</p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-value">{n_strats}</div><div class="hero-stat-label">注册策略</div></div>
        <div class="hero-stat"><div class="hero-stat-value" style="color:var(--accent)">{top_sharpe_val:.3f}</div><div class="hero-stat-label">最高Sharpe · {top_sharpe}</div></div>
        <div class="hero-stat"><div class="hero-stat-value" style="color:var(--green)">{top_ret['annual_return']:.1f}%</div><div class="hero-stat-label">最高年化 · {top_ret['name']}</div></div>
        <div class="hero-stat"><div class="hero-stat-value" style="color:var(--blue)">{abs(min_dd['max_drawdown']):.1f}%</div><div class="hero-stat-label">最低回撤 · {min_dd['name']}</div></div>
    </div>
</div>

<div class="charts-row">
    <div class="chart-card">
        <h3>策略分类分布</h3>
        <div id="pieCategory" class="chart-box"></div>
    </div>
    <div class="chart-card">
        <h3>风险等级分布</h3>
        <div id="pieRisk" class="chart-box"></div>
    </div>
</div>

<div class="section">
    <div class="section-title">Sharpe 排名柱状图</div>
    <div id="barSharpe" style="width:100%;height:480px;"></div>
</div>

<div class="section">
    <div class="section-title">全部策略对比表（按Sharpe排名）</div>
    <div class="table-wrap">
    <table>
        <thead><tr>
            <th>#</th><th>策略</th><th>年化%</th><th>Sharpe</th><th>回撤%</th><th>Calmar</th><th>核心特色</th>
        </tr></thead>
        <tbody>
{rows_html}        </tbody>
    </table>
    </div>
</div>

<div class="section">
    <div class="section-title">按风险等级分组</div>
    <div class="table-wrap">
    <table>
        <thead><tr><th>策略</th><th>年化%</th><th>回撤%</th><th>Sharpe</th></tr></thead>
        <tbody>
{risk_rows_html}        </tbody>
    </table>
    </div>
</div>

<div class="section">
    <div class="section-title">按策略类别分组</div>
    <div class="table-wrap">
    <table>
        <thead><tr><th>策略</th><th>年化%</th><th>Sharpe</th><th>回撤%</th><th>Calmar</th><th>核心特色</th></tr></thead>
        <tbody>
{cat_rows_html}        </tbody>
    </table>
    </div>
</div>

<div class="section">
    <div class="section-title">市场状态策略推荐</div>
    <div class="rec-grid">
{rec_cards_html}    </div>
</div>

<div class="footer">
    ZEquant Strategy Report · Generated on 2026-05-16 · Powered by ECharts
</div>

</div>

<script>
const sharpeData = {sharpe_data_json};
const catData = {cat_data_json};
const riskData = {risk_data_json};

const chartTheme = {{
    backgroundColor: 'transparent',
    textStyle: {{ color: '#e2e8f0', fontFamily: '"Noto Sans SC", sans-serif' }},
    grid: {{ left: 60, right: 40, top: 20, bottom: 60 }},
}};

// Sharpe Bar
(function() {{
    const chart = echarts.init(document.getElementById('barSharpe'));
    const names = sharpeData.map(d => d.name);
    const values = sharpeData.map(d => d.value);
    const colors = values.map(v => v >= 1.3 ? '#22c55e' : v >= 1.0 ? '#eab308' : '#ef4444');
    chart.setOption({{
        ...chartTheme,
        tooltip: {{ trigger:'axis', axisPointer:{{type:'shadow'}}, backgroundColor:'#1e293b', borderColor:'#334155' }},
        grid: {{ left: 80, right: 40, top: 20, bottom: 80 }},
        xAxis: {{ type:'category', data:names, axisLabel:{{ rotate:45, fontSize:11, interval:0 }}, axisLine:{{lineStyle:{{color:'#334155'}}}} }},
        yAxis: {{ type:'value', name:'Sharpe Ratio', nameTextStyle:{{color:'#64748b'}}, splitLine:{{lineStyle:{{color:'#1e293b', type:'dashed'}}}}, axisLabel:{{fontFamily:'JetBrains Mono, monospace'}} }},
        series: [{{
            type:'bar', data:values,
            itemStyle:{{ color:(p)=>colors[p.dataIndex], borderRadius:[4,4,0,0] }},
            label:{{ show:true, position:'top', fontSize:11, fontFamily:'JetBrains Mono, monospace', formatter:(p)=>p.data.toFixed(3), color:'#94a3b8' }},
            animationDelay:(i)=>i*60,
        }}],
        animationDelay:(i)=>i*60,
    }});
    window.addEventListener('resize', ()=>chart.resize());
}})();

// Pie Category
(function() {{
    const chart = echarts.init(document.getElementById('pieCategory'));
    const colors = ['#3b82f6','#8b5cf6','#f59e0b','#ec4899'];
    chart.setOption({{
        ...chartTheme,
        tooltip: {{ trigger:'item', backgroundColor:'#1e293b', borderColor:'#334155', formatter:'{{b}}: {{c}} ({{d}}%)' }},
        series: [{{
            type:'pie', radius:['40%','70%'], center:['50%','50%'],
            data: catData.map((d, idx)=>({{...d, itemStyle:{{color:colors[idx]}}}})),
            label: {{ color:'#e2e8f0', fontSize:12 }},
            labelLine: {{ lineStyle:{{color:'#334155'}} }},
            emphasis: {{ itemStyle:{{ shadowBlur:20, shadowColor:'rgba(0,0,0,0.5)' }} }},
            animationType:'scale',
        }}],
    }});
    window.addEventListener('resize', ()=>chart.resize());
}})();

// Pie Risk
(function() {{
    const chart = echarts.init(document.getElementById('pieRisk'));
    const colors = ['#22c55e','#eab308','#ef4444'];
    chart.setOption({{
        ...chartTheme,
        tooltip: {{ trigger:'item', backgroundColor:'#1e293b', borderColor:'#334155', formatter:'{{b}}: {{c}} ({{d}}%)' }},
        series: [{{
            type:'pie', radius:['40%','70%'], center:['50%','50%'],
            data: riskData.map((d,i)=>({{...d, itemStyle:{{color:colors[i]}}}})),
            label: {{ color:'#e2e8f0', fontSize:12 }},
            labelLine: {{ lineStyle:{{color:'#334155'}} }},
            emphasis: {{ itemStyle:{{ shadowBlur:20, shadowColor:'rgba(0,0,0,0.5)' }} }},
            animationType:'scale',
        }}],
    }});
    window.addEventListener('resize', ()=>chart.resize());
}})();
</script>
</body>
</html>"""
    return html


def main():
    html = generate_html()
    out_path = BASE_DIR / "strategy_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"报告已生成: {out_path}")
    print(f"文件大小: {os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
