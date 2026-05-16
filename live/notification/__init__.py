"""
邮件通知模块 — 无券商API，每日操作以邮件发送。

支持两种邮件后端:
  - smtp:    标准SMTP(QQ/163/Gmail)
  - sendmail: 本地sendmail

用法:
    from live.notification.mailer import Mailer
    mailer = Mailer()
    mailer.send_daily_signal("今日调仓清单", orders)
"""
from __future__ import annotations
import os
import logging
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Mailer:
    """邮件发送器 — 支持SMTP和文件输出两种模式。"""

    def __init__(self, config: dict = None):
        self.config = config or self._load_config()

    def _load_config(self) -> dict:
        """从配置文件或环境变量加载邮件配置。"""
        return {
            "backend": os.getenv("MAIL_BACKEND", "file"),  # smtp | file
            "smtp_host": os.getenv("SMTP_HOST", "smtp.qq.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_user": os.getenv("SMTP_USER", ""),
            "smtp_pass": os.getenv("SMTP_PASS", ""),  # 授权码
            "from_addr": os.getenv("MAIL_FROM", ""),
            "to_addr": os.getenv("MAIL_TO", ""),
            "output_dir": os.getenv("MAIL_OUTPUT_DIR", "./data_live/signals"),
        }

    def send_daily_signal(self, subject: str, orders: List[Dict],
                          positions: Dict = None, report: str = None):
        """发送每日调仓信号邮件。

        Args:
            subject: 邮件标题
            orders:  调仓清单 [{symbol, direction, shares, price, reason}, ...]
            positions: 当前持仓 {symbol: shares}
            report:   额外文本报告
        """
        body = self._build_html(orders, positions, report)
        cfg = self.config

        if cfg["backend"] == "smtp":
            self._send_smtp(subject, body, cfg)
        else:
            self._save_to_file(subject, body, orders, cfg)

        logger.info("邮件已发送/保存: %s", subject)

    def _build_html(self, orders: List[Dict], positions: Dict,
                    report: str = None) -> str:
        """构建HTML邮件正文。"""
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        buys = [o for o in orders if o.get("direction", "").upper() in ("BUY", "买入")]
        sells = [o for o in orders if o.get("direction", "").upper() in ("SELL", "卖出", "卖出")]
        holds = list(positions.keys()) if positions else []

        html = f"""
<html><head><meta charset="utf-8"></head><body style="font-family: 'Noto Sans SC', sans-serif; background:#0a0e17; color:#e2e8f0; padding:20px;">
<div style="max-width:680px; margin:0 auto; background:#111827; border-radius:12px; padding:24px; border:1px solid #1e293b;">
<h1 style="color:#f59e0b; font-size:20px; margin:0 0 8px;">ZEquant 每日调仓信号</h1>
<p style="color:#64748b; font-size:13px; margin:0 0 20px;">{today}</p>
"""
        # 买入清单
        html += '<h2 style="color:#ef4444; font-size:16px; margin:16px 0 8px;">🔴 买入</h2>'
        if buys:
            html += '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
            html += '<tr style="background:#1a2332;"><th style="padding:6px 8px; text-align:left; color:#94a3b8;">代码</th><th style="padding:6px 8px; text-align:right; color:#94a3b8;">方向</th><th style="padding:6px 8px; text-align:right; color:#94a3b8;">股数</th><th style="padding:6px 8px; text-align:right; color:#94a3b8;">价格</th><th style="padding:6px 8px; text-align:right; color:#94a3b8;">理由</th></tr>'
            for o in buys:
                html += f'<tr style="border-bottom:1px solid #1e293b;"><td style="padding:6px 8px;">{o.get("symbol","")}</td><td style="padding:6px 8px; text-align:right;">{o.get("direction","")}</td><td style="padding:6px 8px; text-align:right;">{o.get("shares","")}</td><td style="padding:6px 8px; text-align:right;">{o.get("price","—")}</td><td style="padding:6px 8px; text-align:right; color:#64748b;">{o.get("reason","")}</td></tr>'
            html += '</table>'
        else:
            html += '<p style="color:#64748b;">今日无买入信号</p>'

        # 卖出清单
        html += '<h2 style="color:#22c55e; font-size:16px; margin:16px 0 8px;">🟢 卖出</h2>'
        if sells:
            html += '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
            html += '<tr style="background:#1a2332;"><th style="padding:6px 8px; text-align:left; color:#94a3b8;">代码</th><th style="padding:6px 8px; text-align:right; color:#94a3b8;">方向</th><th style="padding:6px 8px; text-align:right; color:#94a3b8;">股数</th><th style="padding:6px 8px; text-align:right; color:#94a3b8;">理由</th></tr>'
            for o in sells:
                html += f'<tr style="border-bottom:1px solid #1e293b;"><td style="padding:6px 8px;">{o.get("symbol","")}</td><td style="padding:6px 8px; text-align:right;">{o.get("direction","")}</td><td style="padding:6px 8px; text-align:right;">{o.get("shares","")}</td><td style="padding:6px 8px; text-align:right; color:#64748b;">{o.get("reason","")}</td></tr>'
            html += '</table>'
        else:
            html += '<p style="color:#64748b;">今日无卖出信号</p>'

        # 当前持仓
        if holds:
            html += '<h2 style="color:#f59e0b; font-size:16px; margin:16px 0 8px;">📦 当前持仓</h2>'
            html += f'<p style="color:#64748b; font-size:13px;">{"、".join(holds[:15])}</p>'

        # 额外报告
        if report:
            html += f'<div style="margin-top:16px; padding:12px; background:#0f1729; border-radius:8px; font-size:13px; color:#94a3b8;"><pre style="white-space:pre-wrap;">{report}</pre></div>'

        html += """
<p style="color:#64748b; font-size:11px; margin-top:24px; text-align:center;">ZEquant — 自动生成，仅供参考，不构成投资建议</p>
</div></body></html>"""
        return html

    def _send_smtp(self, subject: str, html: str, cfg: dict):
        """通过SMTP发送邮件。"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = cfg["from_addr"]
        msg["To"] = cfg["to_addr"]
        msg.attach(MIMEText(html, "html", "utf-8"))

        try:
            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.starttls()
                server.login(cfg["smtp_user"], cfg["smtp_pass"])
                server.sendmail(cfg["from_addr"], cfg["to_addr"], msg.as_string())
            logger.info("SMTP邮件发送成功 -> %s", cfg["to_addr"])
        except Exception as e:
            logger.error("SMTP邮件发送失败: %s", e)
            self._save_to_file(subject, html, [], cfg)

    def _save_to_file(self, subject: str, html: str, orders: List[Dict], cfg: dict):
        """保存到文件（兜底/本地模式）。"""
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        html_path = os.path.join(out_dir, f"signal_{today}.html")
        json_path = os.path.join(out_dir, f"orders_{today}.json")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(orders, f, ensure_ascii=False, indent=2)

        logger.info("信号文件已保存: %s (%d笔订单)", html_path, len(orders))

    @staticmethod
    def format_orders(signals: List[Dict], current_positions: Dict[str, int],
                      cash: float = 0) -> List[Dict]:
        """将信号转为可读调仓清单。

        Args:
            signals: 信号列表 [{symbol, weight, ...}, ...]
            current_positions: {symbol: shares}
            cash: 当前可用资金

        Returns:
            调仓清单 [{symbol, direction, shares, reason}, ...]
        """
        orders = []
        symbols_in_signal = set()

        for sig in signals:
            sym = sig["symbol"]
            symbols_in_signal.add(sym)
            if sym in current_positions:
                continue  # 已有持仓，保持（后续可根据权重调整股数）
            orders.append({
                "symbol": sym,
                "direction": "买入",
                "shares": 100,  # 1手起步
                "price": sig.get("price", "—"),
                "reason": sig.get("reason", "信号选入"),
            })

        # 不在信号的持仓→卖出
        for sym, shares in (current_positions or {}).items():
            if sym not in symbols_in_signal:
                orders.append({
                    "symbol": sym,
                    "direction": "卖出",
                    "shares": shares,
                    "price": "—",
                    "reason": "退出信号池",
                })

        return orders
