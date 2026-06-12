"""每日生产调度器 — 全自动流水线。

每天收盘后（15:30）依次执行：
  1. 增量拉数据
  2. 生成明日信号 + 写入日期目录 + 发送邮件（含绩效和风控）

用法：
    python3 -m live.runner                       # 全流程(自动检测资金)
    python3 -m live.runner --mode quick           # 只出信号+发邮件（不拉数据）
    python3 -m live.runner --mode email           # 只发邮件（重新发送最新信号）
    python3 -m live.runner --capital 50000        # 手动指定资金(覆盖自动检测)
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live.runner")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAPITAL = 0  # 0 = 由信号生成器自动从DB读取


def run_step(desc: str, script_args: list, fatal: bool = False) -> bool:
    """运行一步，返回 True/False。"""
    logger.info("阶段: %s", desc)
    logger.info("  执行: python3 %s", " ".join(script_args))
    try:
        result = subprocess.run(
            [sys.executable] + script_args,
            capture_output=True, text=True, cwd=PROJECT_DIR, timeout=600,
        )
        if result.returncode != 0:
            logger.error("  ❌ 失败 (exit=%d): %s", result.returncode, result.stderr[:200])
            if fatal:
                sys.exit(1)
            return False
        # 捕获关键输出行
        key_patterns = [
            "✅", "❌", "⚠️",
            "信号已写入", "SMTP邮件发送", "邮件已发送",
            "市场状态", "market_state",
            "n_hold", "n_sell", "n_buy",
            "持仓", "子策略", "持有", "占用", "剩余", "资金",
            "📈", "📉",
            "全流程完成",
        ]
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line:
                continue
            if any(p in line for p in key_patterns):
                logger.info("  %s", line)
        return True
    except subprocess.TimeoutExpired:
        logger.error("  ❌ 超时 (300s)")
        return False
    except Exception as e:
        logger.error("  ❌ 异常: %s", e)
        return False


def _check_hard_stop():
    """检查实盘累计回撤是否超过 -10% 硬止损线。"""
    try:
        from core.database import Database
        db = Database("./data_live/live_data.db")
        row = db.conn.execute(
            "SELECT cumulative FROM daily_performance ORDER BY date DESC LIMIT 1"
        ).fetchone()
        db.close()
        if row:
            cum = float(row[0])
            dd = cum - 1.0  # cumulative 从 1.0 开始
            if dd < -0.10:
                logger.error("❌ 硬止损触发! 累计回撤 %.2f%% <= -10%%, 阻断当日信号", dd * 100)
                sys.exit(1)
            elif dd < -0.08:
                logger.warning("⚠️  回撤观察线: 累计回撤 %.2f%% 接近 -10%% 硬止损", dd * 100)
    except Exception as e:
        logger.warning("无法检查硬止损(非关键): %s", e)


def run_full(capital: float, extra_args: list, skip_data: bool = False):
    """全流程：拉数据 → 信号 → 邮件。

    Args:
        capital: 资金
        extra_args: 额外参数
        skip_data: 是否跳过数据更新
    """
    # 硬止损检查：实盘累计回撤 > 10% 时阻断
    _check_hard_stop()

    # 数据更新
    if not skip_data:
        data_ok = run_step("数据更新", ["-m", "live.data_updater"], fatal=False)
        if not data_ok:
            logger.warning("⚠️  数据更新失败, 跳过当天信号生成 (阻断)")
            return

    sig_args = ["-m", "live.signals.mss_dynamic", "--capital", str(capital), "--email"] + extra_args
    ok = run_step("信号生成", sig_args, fatal=True)
    if not ok:
        return

    logger.info("✅ 全流程完成")


def run_email_only(capital: float):
    """只发邮件（重新发送最新信号）。"""
    from live.notification import Mailer
    from live.performance.tracker import load_latest_signal

    sig = load_latest_signal()
    if sig:
        sig_date = sig.get("meta", {}).get("signal_date", "")
        sig_dir = f"data_live/mss_dynamic/{sig_date.replace('-', '')}"
        sig_path = os.path.join(sig_dir, "build.json")
        if os.path.exists(sig_path):
            mailer = Mailer()
            mailer.send_signal_from_file(sig_path)
            logger.info("✅ 邮件已发送")
        else:
            logger.error("信号文件不存在: %s", sig_path)
    else:
        logger.error("未找到信号文件")


def main():
    parser = argparse.ArgumentParser(description="ZEquant 每日调度器")
    parser.add_argument("--mode", default="full", choices=["full", "quick", "email"],
                        help="full=全流程 quick=信号+邮件(不拉数据) email=只发邮件")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="资金(默认0=自动从DB读取)")
    parser.add_argument("--force", action="store_true", help="强制调仓")
    parser.add_argument("--date", help="指定日期")
    args = parser.parse_args()

    extra = []
    if args.force:
        extra.append("--force")
    if args.date:
        extra.extend(["--date", args.date])

    if args.mode == "email":
        run_email_only(args.capital)
    elif args.mode == "quick":
        run_full(args.capital, extra, skip_data=True)
    else:
        run_full(args.capital, extra, skip_data=False)


if __name__ == "__main__":
    main()
