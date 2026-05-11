"""报告生成模块

功能：
- 回测报告生成
- 实盘报告生成
- HTML/PDF/JSON格式导出
- 对比分析报告
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """报告元数据"""
    title: str
    strategy_name: str
    start_date: str
    end_date: str
    generated_at: datetime
    author: str = ""
    version: str = "1.0"


class ReportGenerator:
    """报告生成器

    生成各种格式的回测和实盘报告
    """

    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir

    def generate_backtest_report(
        self,
        report_data: Dict,
        output_path: str,
        format: str = "html",
    ) -> str:
        """生成回测报告

        Args:
            report_data: 回测报告数据
            output_path: 输出文件路径
            format: 输出格式（html/text/json）
        """
        if format == "json":
            return self._generate_json_report(report_data, output_path)
        elif format == "html":
            return self._generate_html_report(report_data, output_path)
        else:
            return self._generate_text_report(report_data, output_path)

    def _generate_json_report(self, data: Dict, output_path: str) -> str:
        """生成JSON报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"JSON报告已生成: {output_file}")
        return str(output_file)

    def _generate_text_report(self, data: Dict, output_path: str) -> str:
        """生成文本报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("=" * 80)
        lines.append("回测报告".center(80))
        lines.append("=" * 80)
        lines.append("")

        if 'strategy_name' in data:
            lines.append(f"策略名称: {data['strategy_name']}")
        if 'start_date' in data and 'end_date' in data:
            lines.append(f"回测区间: {data['start_date']} ~ {data['end_date']}")
        lines.append("")

        lines.append("-" * 80)
        lines.append("收益指标")
        lines.append("-" * 80)
        metrics = [
            ('总收益率', 'total_return', '%'),
            ('年化收益率', 'annualized_return', '%'),
            ('最大回撤', 'max_drawdown', '%'),
            ('夏普比率', 'sharpe_ratio', ''),
            ('索提诺比率', 'sortino_ratio', ''),
            ('卡玛比率', 'calmar_ratio', ''),
            ('胜率', 'win_rate', '%'),
            ('盈亏比', 'profit_factor', ''),
            ('交易次数', 'total_trades', ''),
        ]
        for label, key, unit in metrics:
            if key in data:
                val = data[key]
                if unit == '%':
                    lines.append(f"  {label}: {val*100:+.2f}%")
                else:
                    lines.append(f"  {label}: {val:+.2f}")
        lines.append("")

        if 'equity_curve' in data and data['equity_curve'] is not None:
            df = data['equity_curve']
            if isinstance(df, pd.DataFrame) and 'total_value' in df.columns:
                lines.append("-" * 80)
                lines.append("净值曲线摘要")
                lines.append("-" * 80)
                lines.append(f"  初始净值: {df['total_value'].iloc[0]:,.2f}")
                lines.append(f"  最终净值: {df['total_value'].iloc[-1]:,.2f}")
                lines.append(f"  峰值净值: {df['total_value'].max():,.2f}")
                lines.append(f"  最低净值: {df['total_value'].min():,.2f}")

        lines.append("")
        lines.append("=" * 80)

        content = "\n".join(lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"文本报告已生成: {output_file}")
        return str(output_file)

    def _generate_html_report(self, data: Dict, output_path: str) -> str:
        """生成HTML报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告 - {data.get('strategy_name', 'Unknown')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header .meta {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }}
        .metric .label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metric .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric .value.positive {{
            color: #28a745;
        }}
        .metric .value.negative {{
            color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .chart-placeholder {{
            height: 300px;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            border-radius: 8px;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>回测报告</h1>
        <div class="meta">
            <div>策略: {data.get('strategy_name', 'N/A')}</div>
            <div>区间: {data.get('start_date', 'N/A')} ~ {data.get('end_date', 'N/A')}</div>
            <div>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </div>

    <div class="card">
        <h2>收益指标</h2>
        <div class="metrics">
            <div class="metric">
                <div class="label">总收益率</div>
                <div class="value {'positive' if data.get('total_return', 0) > 0 else 'negative'}">
                    {data.get('total_return', 0)*100:+.2f}%
                </div>
            </div>
            <div class="metric">
                <div class="label">年化收益率</div>
                <div class="value {'positive' if data.get('annualized_return', 0) > 0 else 'negative'}">
                    {data.get('annualized_return', 0)*100:+.2f}%
                </div>
            </div>
            <div class="metric">
                <div class="label">最大回撤</div>
                <div class="value negative">
                    {data.get('max_drawdown', 0)*100:.2f}%
                </div>
            </div>
            <div class="metric">
                <div class="label">夏普比率</div>
                <div class="value">
                    {data.get('sharpe_ratio', 0):.2f}
                </div>
            </div>
            <div class="metric">
                <div class="label">索提诺比率</div>
                <div class="value">
                    {data.get('sortino_ratio', 0):.2f}
                </div>
            </div>
            <div class="metric">
                <div class="label">胜率</div>
                <div class="value">
                    {data.get('win_rate', 0)*100:.1f}%
                </div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>资金概况</h2>
        <table>
            <tr>
                <td>初始本金</td>
                <td style="text-align: right;">{data.get('initial_capital', 0):,.2f}</td>
            </tr>
            <tr>
                <td>期末现金</td>
                <td style="text-align: right;">{data.get('final_cash', 0):,.2f}</td>
            </tr>
            <tr>
                <td>期末持仓</td>
                <td style="text-align: right;">{data.get('final_position_value', 0):,.2f}</td>
            </tr>
            <tr>
                <td>期末总值</td>
                <td style="text-align: right; font-weight: bold;">{data.get('final_value', 0):,.2f}</td>
            </tr>
            <tr>
                <td>绝对盈亏</td>
                <td style="text-align: right; color: {'green' if data.get('final_value', 0) > data.get('initial_capital', 0) else 'red'};">
                    {data.get('final_value', 0) - data.get('initial_capital', 0):+,.2f}
                </td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>净值曲线</h2>
        <div class="chart-placeholder">
            净值曲线图表（需要前端渲染）
        </div>
    </div>

    <div class="footer">
        Generated by ZeQuant | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML报告已生成: {output_file}")
        return str(output_file)

    def generate_comparison_report(
        self,
        reports: List[Dict],
        output_path: str,
        title: str = "策略对比报告",
    ) -> str:
        """生成对比报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, r in enumerate(reports):
            name = r.get('strategy_name', f'策略{i+1}')
            rows.append({
                '策略名称': name,
                '总收益率': f"{r.get('total_return', 0)*100:+.2f}%",
                '年化收益': f"{r.get('annualized_return', 0)*100:+.2f}%",
                '最大回撤': f"{r.get('max_drawdown', 0)*100:.2f}%",
                '夏普比率': f"{r.get('sharpe_ratio', 0):.2f}",
                '盈亏比': f"{r.get('profit_factor', 0):.2f}",
            })

        df = pd.DataFrame(rows)
        lines = []
        lines.append("=" * 100)
        lines.append(title.center(100))
        lines.append("=" * 100)
        lines.append("")
        lines.append(df.to_string(index=False))
        lines.append("")
        lines.append("=" * 100)

        content = "\n".join(lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"对比报告已生成: {output_file}")
        return str(output_file)
