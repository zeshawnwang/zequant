"""
邮件通知模块 — V6 版本，支持 SMTP 和文件输出。

配置从 .env 或环境变量加载:
  MAIL_BACKEND  SMTP_HOST  SMTP_PORT  SMTP_USER  SMTP_PASS  MAIL_FROM  MAIL_TO

用法:
    from live.notification import Mailer
    mailer = Mailer()
    mailer.send_signal_from_file("data_live/mss_dynamic/20260527/build.json")
"""
from __future__ import annotations
import os
import logging
import smtplib
import ssl
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)


def _next_trading_day(d: str | date) -> tuple[str, str]:
    _WD = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    if isinstance(d, str):
        dt = date.fromisoformat(d)
    else:
        dt = d
    while True:
        dt += timedelta(days=1)
        if dt.weekday() < 5:
            return dt.isoformat(), _WD[dt.weekday()]


def _load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip("\"'")
            if key not in os.environ:
                os.environ[key] = val

_load_dotenv()


class Mailer:
    def __init__(self, config: dict = None):
        self.config = config or self._load_config()

    def _load_config(self) -> dict:
        return {
            "backend": os.getenv("MAIL_BACKEND", "file"),
            "smtp_host": os.getenv("SMTP_HOST", "smtp.qq.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_user": os.getenv("SMTP_USER", ""),
            "smtp_pass": os.getenv("SMTP_PASS", ""),
            "from_addr": os.getenv("MAIL_FROM", ""),
            "to_addr": os.getenv("MAIL_TO", ""),
            "output_dir": os.getenv("MAIL_OUTPUT_DIR", "./data_live/signals"),
        }

    def send_signal_from_file(self, signal_path: str):
        if not os.path.exists(signal_path):
            logger.error("信号文件不存在: %s", signal_path)
            return

        with open(signal_path, encoding="utf-8") as f:
            sig = json.load(f)

        m = sig.get("meta", {})
        date_str = m.get("signal_date", "")
        data_date = m.get("data_date", date_str)
        state = m.get("market_state", "未知")
        n_buy = m.get("n_buy", 0)
        n_sell = m.get("n_sell", 0)
        is_hold = m.get("is_hold", False)
        capital = m.get("capital", 0)
        total_cost = m.get("total_cost", 0)
        remain = m.get("remain", 0)
        confidence = m.get("confidence", 0)
        breadth = m.get("breadth")
        mode = m.get("config", "v2b")

        next_date, next_dow = _next_trading_day(date_str)

        rebalanced_names = [
            sd["name"] for sd in sig.get("sub_details", [])
            if sd.get("status") == "rebalanced"
        ]

        state_label_map = {"bull": "多头 🐂", "bear": "空头 🐻", "oscillate": "震荡 〰️", "recovery": "反弹 📈"}
        state_label = state_label_map.get(state, state)
        breadth_str = f" 广度={breadth:.3f}" if breadth is not None else ""

        if is_hold:
            subject = f"[ZEquant] {date_str} 收盘(用于{next_date} {next_dow}) — 持有不动 ({state_label}{breadth_str})"
        else:
            parts = []
            if n_buy: parts.append(f"买{n_buy}只")
            if n_sell: parts.append(f"卖{n_sell}只")
            tag = ",".join(rebalanced_names) if rebalanced_names else "调仓"
            subject = f"[ZEquant] {date_str} 收盘(用于{next_date} {next_dow}) — {tag} ({' '.join(parts)}) ({state_label}{breadth_str})"

        # -- 状态分析文本 (只含状态/广度/风控，不含子策略——子策略在HTML表格中) --
        state_lines = [f"市场: {state_label}  置信度: {(confidence or 0)*100:.0f}%"]
        if breadth is not None:
            state_lines.append(f"市场广度: {breadth:.3f} {'⚠️ 低于0.35已降级' if breadth < 0.35 else '正常'}")
        if m.get("state_changed"):
            state_lines.append(f"状态变化: {m.get('last_state')} → {state}")

        state_analysis = {
            "bull": "🐂 多头市场 — 多因子满仓进攻 (mf 60% + mf_vol 20% + chip_covrp 20%)",
            "bear": "🐻 空头市场 — 筹码防御为主 (chip_covrp 60% + mf_vol 20% + chip_rp 20%)",
            "oscillate": "〰️ 震荡市场 — 均衡配置 (chip_covrp 40% + mf50_chip50 30% + c01_layered_d5 30%)",
            "recovery": "📈 反弹市场 — 超跌反转 + 筹码保护 (chip_covrp 40% + osr_d10 30% + mf_vol 30%)",
        }
        if state in state_analysis:
            state_lines.append(state_analysis[state])

        # -- 绩效 (从 live DB 加载) --
        perf_lines = []
        try:
            from live.performance.tracker import calc_position_pnl
            from core.database import Database
            live_db = Database("./data_live/live_data.db")
            quant_db = Database("./data/quant_data.db")
            pnl = calc_position_pnl(live_db, quant_db, date_str)
            live_db.close(); quant_db.close()
            if pnl.get("positions"):
                winners = sum(1 for p in pnl["positions"].values() if p["pnl"] >= 0)
                losers = len(pnl["positions"]) - winners
                perf_lines.append(f"总盈亏: {pnl['total_pnl']:+,.0f} ({pnl['total_pnl_pct']:+.2f}%)")
                perf_lines.append(f"盈利 {winners} 只 / 亏损 {losers} 只")
                danger = [s for s, p in pnl["positions"].items() if p["pnl_pct"] < -10]
                if danger:
                    perf_lines.append(f"⚠️ 风控预警: {', '.join(danger)} 亏损 > 10%!")
        except Exception as e:
            logger.debug("绩效加载跳过: %s", e)

        state_text = "\n".join(state_lines)
        perf_text = "\n".join(perf_lines) if perf_lines else ""

        # -- 加载股票名称 --
        names = sig.get("names", {})
        all_symbols = set()
        for o in sig.get("buy_orders", []):
            all_symbols.add(o.get("symbol", ""))
        for o in sig.get("sell_orders", []):
            all_symbols.add(o.get("symbol", ""))
        if not names and all_symbols:
            names = _lookup_names(list(all_symbols))

        meta = {
            "signal_date": date_str, "data_date": data_date,
            "next_trade_date": next_date, "next_trade_dow": next_dow,
            "state": state, "state_label": state_label,
            "confidence": confidence, "breadth": breadth,
            "capital": capital, "total_cost": total_cost, "remain": remain,
            "is_hold": is_hold, "mode": mode, "n_buy": n_buy, "n_sell": n_sell,
        }
        self.send_daily_signal(subject,
                               sig.get("buy_orders", []) + sig.get("sell_orders", []),
                               names, sig.get("sub_details", []),
                               state_text, perf_text, meta)

    def send_daily_signal(self, subject: str, orders: List[Dict],
                          names: Dict[str, str] = None,
                          sub_details: List[Dict] = None,
                          state_text: str = None, perf_text: str = None,
                          meta: Dict = None):
        body = self._build_html(orders, names or {}, sub_details or [],
                                state_text, perf_text, meta or {})
        cfg = self.config
        if cfg["backend"] == "smtp":
            self._send_smtp(subject, body, cfg)
        else:
            self._save_to_file(subject, body, orders, cfg)
        logger.info("邮件已发送/保存: %s", subject)

    def _build_html(self, orders: List[Dict], names: Dict[str, str],
                    sub_details: List[Dict],
                    state_text: str, perf_text: str,
                    meta: Dict) -> str:
        buys = [o for o in orders if o.get("direction", "") in ("BUY", "B", "买入")]
        sells = [o for o in orders if o.get("direction", "") in ("SELL", "S", "卖出")]

        signal_date = meta.get("signal_date", "")
        data_date = meta.get("data_date", "")
        next_trade_date = meta.get("next_trade_date", "")
        next_trade_dow = meta.get("next_trade_dow", "")
        state = meta.get("state", "")
        state_label = meta.get("state_label", state)
        confidence = meta.get("confidence", 0)
        capital = meta.get("capital", 0)
        total_cost = meta.get("total_cost", 0)
        remain = meta.get("remain", 0)
        breadth = meta.get("breadth")
        is_hold = meta.get("is_hold", False)
        mode = meta.get("mode", "v2b")

        # -- 子策略表格 --
        sub_rows = ""
        status_icons = {"rebalanced": "🔄", "hold": "✅", "liquidated": "❌", "skip_empty": "⚠️"}
        for sd in sub_details:
            icon = status_icons.get(sd["status"], "❓")
            if sd["status"] == "rebalanced":
                detail = f"买{sd.get('n_buy',0)}只 卖{sd.get('n_sell',0)}只"
            elif sd["status"] == "hold":
                detail = f"持仓{sd.get('n_hold',0)}只"
                nr = sd.get("next_rebalance")
                if nr: detail += f" · 下次调仓≈{nr}"
            elif sd["status"] == "liquidated":
                detail = f"清仓{sd.get('n_sell',0)}只"
            else:
                detail = ""
            sub_rows += f'<tr><td style="padding:3px 0;font-size:13px;color:#555;">{icon} {sd["name"]} — {detail}</td></tr>'

        # -- 买入表格 --
        buy_rows = ""
        for o in buys:
            nm = names.get(o.get("symbol", ""), "")
            buy_rows += f"""<tr>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;">{o.get("symbol","")}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;">{nm}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;text-align:right;">{o.get("shares","")}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;text-align:right;">{o.get("price","—")}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;text-align:right;">{o.get("cost",0):,.0f}</td>
            </tr>"""

        # -- 卖出表格 --
        sell_rows = ""
        for o in sells:
            nm = names.get(o.get("symbol", ""), "")
            sell_rows += f"""<tr>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;">{o.get("symbol","")}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;">{nm}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;text-align:right;">{o.get("shares","")}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;text-align:right;">{o.get("price","—")}</td>
              <td style="padding:7px 6px;border-bottom:1px solid #eee;">{o.get("reason","")}</td>
            </tr>"""

        state_color = {"bull": "#16a34a", "bear": "#dc2626", "oscillate": "#ea580c", "recovery": "#2563eb"}

        # -- 发送时间 --
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table align="center" cellpadding="0" cellspacing="0" style="max-width:640px;width:100%;margin:24px auto;background:#fff;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,0.08);">
<tr><td style="padding:20px 28px 8px;">
  <h1 style="font-size:20px;font-weight:600;color:#1a1a2e;margin:0 0 2px;">ZEquant V6 · 每日信号</h1>
  <p style="font-size:12px;color:#aaa;margin:0 0 14px;">发送时间 {now_str}</p>

  <table cellpadding="0" cellspacing="0" style="width:100%;background:#f8f9fb;border-radius:6px;padding:12px 16px;font-size:13px;color:#333;">
    <tr>
      <td style="padding:2px 0;"><span style="color:#888;">信号日</span><br><strong>{signal_date}</strong></td>
      <td style="padding:2px 0;"><span style="color:#888;">数据截止</span><br><strong>{data_date}</strong></td>
      <td style="padding:2px 0;"><span style="color:#888;">次日开盘</span><br><strong>{next_trade_date} {next_trade_dow}</strong></td>
      <td style="padding:2px 0;text-align:right;">
        <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:13px;font-weight:600;color:#fff;background:{state_color.get(state,'#888')};">{state_label}</span>
      </td>
    </tr>
  </table>

  <table cellpadding="0" cellspacing="0" style="width:100%;margin-top:10px;font-size:13px;color:#555;">
    <tr>
      <td style="padding:4px 0;">资金 <strong style="color:#1a1a2e;">{capital:,.0f}</strong></td>
      <td style="padding:4px 0;">占用 <strong style="color:#1a1a2e;">{total_cost:,.0f}</strong></td>
      <td style="padding:4px 0;">余额 <strong style="color:#1a1a2e;">{remain:,.0f}</strong></td>
      <td style="padding:4px 0;text-align:right;">置信度 <strong style="color:#1a1a2e;">{(confidence or 0)*100:.0f}%</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="font-size:11px;color:#999;padding-top:2px;">rhythm=composite(trend×0.6+vol×0.4) 止盈=3%</td>
      <td colspan="2" style="font-size:11px;color:#999;padding-top:2px;text-align:right;">Breadth: {(breadth or 0):.3f} | {'⚠️降级' if breadth and breadth < 0.35 else '正常'}</td>
    </tr>
  </table>
</td></tr>
"""

        if is_hold and not buys and not sells:
            html += f"""<tr><td style="padding:16px 28px;text-align:center;">
  <div style="display:inline-block;padding:10px 32px;background:#ecfdf5;border:1px solid #bbf7d0;border-radius:8px;font-size:16px;font-weight:600;color:#16a34a;">✅ 今日无操作 · 继续持有</div>
</td></tr>"""

        if sub_details:
            html += f"""<tr><td style="padding:0 28px 12px;">
  <h2 style="font-size:15px;font-weight:600;color:#1a1a2e;margin:0 0 8px;">📊 子策略</h2>
  <table cellpadding="0" cellspacing="0" style="width:100%;font-size:13px;">{sub_rows}</table>
</td></tr>"""

        # 绩效 (固定位置, 在子策略下方)
        if perf_text:
            html += f"""<tr><td style="padding:0 28px 12px;">
  <h2 style="font-size:15px;font-weight:600;color:#1a1a2e;margin:0 0 6px;">📊 持仓绩效</h2>
  <div style="padding:10px 14px;background:#fefce8;border:1px solid #fde68a;border-radius:6px;font-size:13px;color:#555;line-height:1.7;white-space:pre-wrap;">{perf_text}</div>
</td></tr>"""

        # 状态分析
        if state_text:
            html += f"""<tr><td style="padding:0 28px 12px;">
  <div style="padding:10px 14px;background:#f0f4ff;border-radius:6px;font-size:13px;color:#555;line-height:1.7;white-space:pre-wrap;">{state_text}</div>
</td></tr>"""

        if buys:
            html += f"""<tr><td style="padding:0 28px 12px;">
  <h2 style="font-size:15px;font-weight:600;color:#16a34a;margin:0 0 8px;">📈 买入 ({len(buys)}只)</h2>
  <table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead><tr style="background:#f5f6f8;">
      <th style="padding:8px 6px;text-align:left;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">代码</th>
      <th style="padding:8px 6px;text-align:left;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">名称</th>
      <th style="padding:8px 6px;text-align:right;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">股数</th>
      <th style="padding:8px 6px;text-align:right;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">价格</th>
      <th style="padding:8px 6px;text-align:right;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">金额</th>
    </tr></thead>
    <tbody>{buy_rows}</tbody>
  </table>
</td></tr>"""

        if sells:
            html += f"""<tr><td style="padding:0 28px 12px;">
  <h2 style="font-size:15px;font-weight:600;color:#dc2626;margin:0 0 8px;">📉 卖出 ({len(sells)}只)</h2>
  <table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead><tr style="background:#f5f6f8;">
      <th style="padding:8px 6px;text-align:left;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">代码</th>
      <th style="padding:8px 6px;text-align:left;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">名称</th>
      <th style="padding:8px 6px;text-align:right;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">股数</th>
      <th style="padding:8px 6px;text-align:right;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">参考价</th>
      <th style="padding:8px 6px;text-align:left;font-weight:600;color:#555;border-bottom:1px solid #e0e0e0;">理由</th>
    </tr></thead>
    <tbody>{sell_rows}</tbody>
  </table>
</td></tr>"""

        if buys or sells:
            html += f"""<tr><td style="padding:0 28px 10px;">
  <table cellpadding="0" cellspacing="0" style="width:100%;font-size:14px;color:#1a1a2e;">
    <tr><td style="padding:8px 0;border-top:2px solid #e0e0e0;">
      总占用: <strong>{total_cost:,.0f}</strong> | 可用余额: <strong>{remain:,.0f}</strong>
    </td></tr>
  </table>
</td></tr>"""

        html += """<tr><td style="padding:14px 28px 18px;text-align:center;font-size:11px;color:#aaa;border-top:1px solid #eee;">
  ZEquant V6 — 自动生成，仅供参考，不构成投资建议
</td></tr></table></body></html>"""
        return html

    def _send_smtp(self, subject: str, html: str, cfg: dict):
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = cfg["from_addr"]
        msg["To"] = cfg["to_addr"]
        msg.attach(MIMEText(html, "html", "utf-8"))
        port = cfg["smtp_port"]
        try:
            if port == 465:
                with smtplib.SMTP_SSL(cfg["smtp_host"], port, context=ssl.create_default_context()) as server:
                    server.login(cfg["smtp_user"], cfg["smtp_pass"])
                    server.sendmail(cfg["from_addr"], cfg["to_addr"], msg.as_string())
            else:
                with smtplib.SMTP(cfg["smtp_host"], port) as server:
                    server.starttls()
                    server.login(cfg["smtp_user"], cfg["smtp_pass"])
                    server.sendmail(cfg["from_addr"], cfg["to_addr"], msg.as_string())
            logger.info("SMTP邮件发送成功 -> %s", cfg["to_addr"])
        except Exception as e:
            logger.error("SMTP邮件发送失败: %s", e)
            self._save_to_file(subject, html, [], cfg)

    def _save_to_file(self, subject: str, html: str, orders: List[Dict], cfg: dict):
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


def _lookup_names(symbols: List[str]) -> Dict[str, str]:
    cache = {}
    need = [s for s in symbols if s not in cache]
    if need:
        try:
            from core.database import Database
            db = Database("./data/quant_data.db", read_only=True)
            ph = ",".join("?" for _ in need)
            rows = db.conn.execute(f"SELECT symbol, name FROM symbols WHERE symbol IN ({ph})", need).fetchall()
            for r in rows: cache[r[0]] = r[1]
            db.close()
        except Exception:
            pass
    return {s: cache.get(s, "") for s in symbols}
