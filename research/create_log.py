#!/usr/bin/env python3
"""
研究日志记录脚本

用法:
    # 创建新的研究日志
    python research/create_log.py --topic "研究主题"

    # 列出所有研究日志
    python research/create_log.py --list

    # 查看最新的研究日志
    python research/create_log.py --latest
"""
import argparse
import datetime
import os
import re
import shutil
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent
LOGS_DIR = RESEARCH_DIR / "logs"
TEMPLATE_FILE = RESEARCH_DIR / "templates" / "research_log_template.md"


def get_next_research_number() -> int:
    """获取下一个研究编号"""
    if not LOGS_DIR.exists():
        return 1

    log_files = list(LOGS_DIR.glob("R*.md"))
    if not log_files:
        return 1

    numbers = []
    for f in log_files:
        match = re.match(r"R(\d+)", f.stem)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers) + 1 if numbers else 1


def create_new_log(topic: str = "") -> Path:
    """创建新的研究日志"""
    today = datetime.date.today()
    research_num = get_next_research_number()

    log_filename = f"R{research_num:03d}_{today}.md"
    log_filepath = LOGS_DIR / log_filename

    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"模板文件不存在: {TEMPLATE_FILE}")

    shutil.copy(TEMPLATE_FILE, log_filepath)

    with open(log_filepath, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("YYYY-MM-DD", str(today))
    content = content.replace("R001", f"R{research_num:03d}")

    if topic:
        content = content.replace("简短描述本次研究主题", topic)

    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return log_filepath


def list_logs(limit: int = 10) -> list:
    """列出研究日志"""
    if not LOGS_DIR.exists():
        return []

    log_files = sorted(LOGS_DIR.glob("R*.md"), reverse=True)

    result = []
    for i, f in enumerate(log_files[:limit]):
        with open(f, "r", encoding="utf-8") as f_in:
            first_line = f_in.readline().strip()

        result.append({
            "file": f.name,
            "path": str(f),
            "topic": first_line.strip("# ") if first_line else ""
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="研究日志管理")
    parser.add_argument("--topic", "-t", help="研究主题", default="")
    parser.add_argument("--list", "-l", action="store_true", help="列出研究日志")
    parser.add_argument("--latest", "-n", action="store_true", help="查看最新日志")
    parser.add_argument("--number", type=int, help="列出数量", default=10)

    args = parser.parse_args()

    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        logs = list_logs(args.number)
        if not logs:
            print("暂无研究日志")
            return

        print("=" * 80)
        print("研究日志列表")
        print("=" * 80)
        for log in logs:
            print(f"{log['file']} - {log['topic']}")
        return

    if args.latest:
        logs = list_logs(1)
        if not logs:
            print("暂无研究日志")
            return

        log_path = logs[0]["path"]
        print(f"最新日志: {logs[0]['file']}")
        print("=" * 80)

        if os.name == "posix":
            os.system(f"open '{log_path}'")
        elif os.name == "nt":
            os.startfile(log_path)
        else:
            print(f"请手动打开: {log_path}")
        return

    log_path = create_new_log(args.topic)
    print(f"研究日志已创建: {log_path}")

    if os.name == "posix":
        os.system(f"open '{log_path}'")
    elif os.name == "nt":
        os.startfile(log_path)


if __name__ == "__main__":
    main()
