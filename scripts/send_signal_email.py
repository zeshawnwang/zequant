"""发送最新实盘信号到邮箱。

自动查找 data_live/mss_dynamic/ 下最新日期目录的信号文件并发送邮件。

用法：
    python3 scripts/send_signal_email.py                  # 发送最新信号
    python3 scripts/send_signal_email.py --strategy mss_dynamic
    python3 scripts/send_signal_email.py --path data_live/mss_dynamic/20260519/build.json
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live.notification import Mailer

logger = logging.getLogger("send_signal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STRATEGY_DIRS = {
    "mss_dynamic": "data_live/mss_dynamic",
    "mf_d10_rp": "data_live/mf_d10_rp",
}


def find_latest_signal(strategy: str) -> str:
    """查找策略下最新日期目录的 build.json。"""
    base = STRATEGY_DIRS.get(strategy, f"data_live/{strategy}")
    if not os.path.exists(base):
        logger.error("策略目录不存在: %s", base)
        return None

    dirs = sorted([d for d in os.listdir(base)
                   if d.isdigit() and os.path.isdir(os.path.join(base, d))],
                  reverse=True)
    if not dirs:
        logger.error("未找到任何日期目录: %s", base)
        return None

    for d in dirs:
        p = os.path.join(base, d, "build.json")
        if os.path.exists(p):
            return p

    logger.error("未找到任何信号文件")
    return None


def main():
    parser = argparse.ArgumentParser(description="发送最新实盘信号到邮箱")
    parser.add_argument("--strategy", default="mss_dynamic", help="策略名")
    parser.add_argument("--path", help="直接指定信号JSON路径")
    args = parser.parse_args()

    signal_path = args.path
    if not signal_path:
        signal_path = find_latest_signal(args.strategy)

    if not signal_path:
        sys.exit(1)

    logger.info("发送信号: %s", signal_path)
    mailer = Mailer()
    mailer.send_signal_from_file(signal_path)


if __name__ == "__main__":
    main()
