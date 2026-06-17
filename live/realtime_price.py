"""实时行情获取 — 多数据源兜底。

三个数据源依次尝试，第一个成功返回的即被采用：
  1. akshare  — 全市场快照（单次请求可查多只，最推荐）
  2. 新浪财经 — HTTP 接口 hq.sinajs.cn
  3. 腾讯财经 — HTTP 接口

用法：
    python3 -m live.realtime_price 002416,002552
    python3 -m live.realtime_price
"""
from __future__ import annotations
import json
import logging
import os
import re
import sys
import time
from typing import Dict, Optional

logger = logging.getLogger("realtime_price")

SINA_URL = "https://hq.sinajs.cn/list={}"
TENCENT_URL = "https://qt.gtimg.cn/q={}"


def _market_prefix(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("6", "9")):
        return f"sh{s}"
    return f"sz{s}"


def _sina_suffix(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("6", "9")):
        return f"sh{s}"
    return f"sz{s}"


def _tencent_suffix(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("6", "9")):
        return f"sh{s}"
    return f"sz{s}"


def fetch_akshare(symbols: list) -> Dict[str, float]:
    """数据源 1: akshare 全市场快照。

    优势：一次请求全市场，按股票池过滤即可。
    交易时间返回最新成交价，非交易时间返回收盘价（有标识）。
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare 未安装")
        return {}

    try:
        df = ak.stock_zh_a_spot_em()
        if df.empty:
            return {}

        # 过滤持仓股票（代码列格式可能有前缀或不带前缀）
        code_col = "代码" if "代码" in df.columns else "symbol"
        price_col = "最新价" if "最新价" in df.columns else "current"
        result = {}
        target_set = {s.zfill(6) for s in symbols}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            raw = code.zfill(6) if code.isdigit() else code[-6:].zfill(6)
            if raw in target_set:
                p = float(row[price_col]) if row[price_col] else 0
                if p > 0:
                    result[raw] = p
        if result:
            logger.info(f"  [akshare] 获取 {len(result)} 只实时价成功")
            return result
    except Exception as e:
        logger.warning(f"  [akshare] 失败: {e}")

    return {}


def fetch_sina(symbols: list) -> Dict[str, float]:
    """数据源 1: 新浪财经 HTTP 接口。

    单次请求，逗号分隔多只股票。
    返回格式: "var hq_str_sh600000=名称,今开,昨收,当前价,..."
    轻量级，按需只请求需要的股票。
    """
    try:
        import urllib.request
    except ImportError:
        return {}

    codes = ",".join(_sina_suffix(s) for s in symbols)
    url = SINA_URL.format(codes)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("gbk", errors="ignore")

        result = {}
        target_set = {s.zfill(6) for s in symbols}
        for line in text.strip().split("\n"):
            line = line.strip()
            if "=" not in line or '"' not in line:
                continue
            # var hq_str_sz002416="名称,今开,昨收,当前,..."
            var_name = line.split("=")[0].strip()
            raw_code = var_name.replace("var hq_str_", "").strip()
            code6 = raw_code[-6:].zfill(6)
            if code6 not in target_set:
                continue
            # 提取引号内的值，取第4个字段（当前价，索引3）
            val = line.split('"')[1] if '"' in line else ""
            parts = val.split(",")
            if len(parts) >= 4:
                price_str = parts[3].strip()
                if price_str and price_str != "0.00":
                    result[code6] = float(price_str)
        if result:
            logger.info(f"  [sina] 获取 {len(result)} 只实时价成功")
            return result
    except Exception as e:
        logger.warning(f"  [sina] 失败: {e}")

    return {}


def fetch_tencent(symbols: list) -> Dict[str, float]:
    """数据源 2: 腾讯财经 HTTP 接口。

    单次请求，分号分隔多只股票。
    返回格式: "v_sz002416=...~...~代码~当前价~..."
    轻量级，按需只请求需要的股票。
    """
    try:
        import urllib.request
    except ImportError:
        return {}

    codes = ";".join(_tencent_suffix(s) for s in symbols)
    url = TENCENT_URL.format(codes)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("gbk", errors="ignore")

        result = {}
        target_set = {s.zfill(6) for s in symbols}
        for line in text.strip().split("\n"):
            line = line.strip()
            parts = line.split("~")
            if len(parts) < 4:
                continue
            # 腾讯格式: parts[2] 是代码, parts[3] 是当前价
            code_raw = parts[2].strip()
            code6 = code_raw.zfill(6) if code_raw.isdigit() else code_raw[-6:].zfill(6)
            if code6 not in target_set:
                continue
            price_str = parts[3].strip()
            if price_str and price_str != "0.00":
                result[code6] = float(price_str)
        if result:
            logger.info(f"  [tencent] 获取 {len(result)} 只实时价成功")
            return result
    except Exception as e:
        logger.warning(f"  [tencent] 失败: {e}")

    return {}


FETCHERS = [fetch_sina, fetch_tencent, fetch_akshare]


def get_realtime_prices(symbols: list) -> Dict[str, float]:
    """多源兜底获取实时行情价。

    Args:
        symbols: 股票代码列表（如 ["002416", "002552"]）

    Returns:
        {symbol: price} 字典，只包含成功获取到的股票
    """
    symbols = [s.zfill(6) for s in symbols]
    for fetcher in FETCHERS:
        result = fetcher(symbols)
        if result:
            return result
        time.sleep(0.3)
    logger.warning("所有数据源全部失败，无法获取实时行情")
    return {}


def check_is_trading_time() -> bool:
    """检查当前是否为 A 股盘中交易时间（粗略判断）。

    周一至周五 09:30-11:30, 13:00-15:00。
    节假日不做精确判断（函数名含"粗略"）。
    """
    now = time.localtime()
    # 周末
    if now.tm_wday >= 5:
        return False
    hour, minute = now.tm_hour, now.tm_min
    minutes = hour * 60 + minute
    am_start = 9 * 60 + 30
    am_end = 11 * 60 + 30
    pm_start = 13 * 60
    pm_end = 15 * 60
    return (am_start <= minutes <= am_end) or (pm_start <= minutes <= pm_end)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = sys.argv[1:]
    if args:
        symbols = [s.strip().zfill(6) for s in args[0].split(",") if s.strip()]
    else:
        # 默认：从 sub_strategy_state 读持仓
        from core.database import Database

        live_db = Database("./data_live/live_data.db")
        rows = live_db.conn.execute(
            "SELECT holdings FROM sub_strategy_state"
        ).fetchall()
        live_db.close()
        symbols = []
        for r in rows:
            h = json.loads(r[0]) if r[0] else {}
            symbols.extend(h.keys())
        symbols = sorted(set(symbols))

    if not symbols:
        logger.warning("无持仓股票，退出")
        return

    logger.info(f"查询 {len(symbols)} 只: {', '.join(symbols)}")
    prices = get_realtime_prices(symbols)
    if prices:
        for sym in sorted(prices):
            logger.info(f"  {sym}: {prices[sym]:.3f}")
    else:
        logger.warning("获取失败")


if __name__ == "__main__":
    main()
