#!/usr/bin/env python3
"""Fama-French 因子计算脚本。

使用 AKShare 免费数据计算 A股适配版 Fama-French 因子:
- MKT: 市场因子 (沪深300收益 - 无风险利率)
- SMB: 规模因子 (小市值 - 大市值)
- HML: 价值因子 (1/PB 代替账面市值比)
- Size: 市值对数 (横截面因子)
- Value: PB倒数 (横截面因子)

示例:
    python3 scripts/compute_fama_french.py --all
    python3 scripts/compute_fama_french.py --names ff_mkt ff_smb ff_hml
    python3 scripts/compute_fama_french.py --start 2024-01-01
"""
from __future__ import annotations
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import load_config, get_db_path
from core.database import Database
from core.factor_hub import FactorHub
import factors.fama_french


def _get_akshare_data(symbols: list, start_date: str, end_date: str) -> dict:
    """从 AKShare 获取市值和 PB 数据。

    Returns:
        dict: {
            "market_cap": DataFrame(index=date, columns=symbol),
            "pb": DataFrame(index=date, columns=symbol),
        }
    """
    try:
        import akshare as ak
    except ImportError:
        print("错误: 需要安装 akshare: pip install akshare")
        return {"market_cap": pd.DataFrame(), "pb": pd.DataFrame()}

    market_caps = {}
    pb_ratios = {}

    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")

    for sym in symbols:
        try:
            basic = ak.stock_zh_a_daily_basic(
                symbol=sym,
                start_date=start_str,
                end_date=end_str,
            )
            if basic is not None and not basic.empty:
                if "total_market_cap" in basic.columns:
                    market_caps[sym] = basic.set_index("date")["total_market_cap"] / 1e8
                elif "total_share" in basic.columns and "close" in basic.columns:
                    mcap = basic.set_index("date")["total_share"] * basic.set_index("date")["close"] / 1e8
                    market_caps[sym] = mcap
                if "pb" in basic.columns:
                    pb_ratios[sym] = basic.set_index("date")["pb"]
        except Exception:
            continue

    mcap_df = pd.DataFrame(market_caps) if market_caps else pd.DataFrame()
    pb_df = pd.DataFrame(pb_ratios) if pb_ratios else pd.DataFrame()

    return {
        "market_cap": mcap_df,
        "pb": pb_df,
    }


def _get_market_return_data() -> pd.Series:
    """获取沪深300日收益率。"""
    try:
        import akshare as ak
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        hs300["returns"] = hs300["close"].pct_change()
        hs300.index = pd.to_datetime(hs300["date"])
        return hs300["returns"]
    except Exception as e:
        print(f"警告: 无法获取沪深300数据: {e}")
        return pd.Series()


def _get_risk_free_rate() -> pd.Series:
    """获取国债收益率作为无风险利率。"""
    try:
        import akshare as ak
        bond = ak.bond_china_treasury_yield()
        rf = bond.set_index("日期")["收益率"] / 252 / 100
        return rf
    except Exception:
        print("警告: 无法获取国债收益率,使用固定无风险利率 0.03")
        return pd.Series()


def compute_mkt(bars: pd.DataFrame) -> pd.Series:
    """计算 MKT 因子(市场超额收益)。"""
    market_ret = _get_market_return_data()
    rf = _get_risk_free_rate()

    mkt = market_ret - rf
    return mkt


def compute_smb(bars: pd.DataFrame, market_cap: pd.DataFrame, returns: pd.DataFrame,
                 lookback: int = 60) -> pd.Series:
    """计算 SMB 因子(规模因子)。

    Args:
        bars: 日线数据
        market_cap: 市值数据
        returns: 收益率数据
        lookback: 市值分组回看窗口

    Returns:
        SMB 时序因子
    """
    if market_cap.empty:
        return pd.Series()

    smb_series = []

    for i in range(lookback, len(market_cap)):
        date = market_cap.index[i]
        prev_mcaps = market_cap.iloc[i-lookback:i].iloc[-1]

        valid = prev_mcaps.dropna()
        valid = valid[valid > 0]
        if len(valid) < 10:
            continue

        try:
            quant33 = valid.quantile(0.33)
            quant66 = valid.quantile(0.66)

            small = valid[valid <= quant33].index.tolist()
            big = valid[valid >= quant66].index.tolist()

            if date in returns.index:
                ret_row = returns.loc[date]
                small_ret = ret_row[small].mean() if small else 0
                big_ret = ret_row[big].mean() if big else 0
                smb_series.append({"date": date, "smb": small_ret - big_ret})
        except Exception:
            continue

    if smb_series:
        result = pd.DataFrame(smb_series).set_index("date")["smb"]
        return result
    return pd.Series()


def compute_hml(bars: pd.DataFrame, pb_ratio: pd.DataFrame, returns: pd.DataFrame,
                 lookback: int = 60) -> pd.Series:
    """计算 HML 因子(价值因子)。

    Args:
        bars: 日线数据
        pb_ratio: PB 数据
        returns: 收益率数据
        lookback: PB 分组回看窗口

    Returns:
        HML 时序因子
    """
    if pb_ratio.empty:
        return pd.Series()

    inv_pb = 1 / pb_ratio.replace(0, np.nan)
    hml_series = []

    for i in range(lookback, len(inv_pb)):
        date = inv_pb.index[i]
        prev_pbs = inv_pb.iloc[i-lookback:i].iloc[-1]

        valid = prev_pbs.dropna()
        valid = valid[valid > 0]
        if len(valid) < 10:
            continue

        try:
            quant33 = valid.quantile(0.33)
            quant66 = valid.quantile(0.66)

            high = valid[valid >= quant66].index.tolist()
            low = valid[valid <= quant33].index.tolist()

            if date in returns.index:
                ret_row = returns.loc[date]
                high_ret = ret_row[high].mean() if high else 0
                low_ret = ret_row[low].mean() if low else 0
                hml_series.append({"date": date, "hml": high_ret - low_ret})
        except Exception:
            continue

    if hml_series:
        result = pd.DataFrame(hml_series).set_index("date")["hml"]
        return result
    return pd.Series()


def compute_size_cross_section(bars: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """计算 Size 横截面因子(市值对数)。

    Returns:
        DataFrame: index=date, columns=symbol
    """
    if market_cap.empty:
        return pd.DataFrame()

    size_factor = np.log(market_cap.replace(0, np.nan) + 1)
    return size_factor


def compute_value_cross_section(bars: pd.DataFrame, pb_ratio: pd.DataFrame) -> pd.DataFrame:
    """计算 Value 横截面因子(PB倒数)。

    Returns:
        DataFrame: index=date, columns=symbol
    """
    if pb_ratio.empty:
        return pd.DataFrame()

    value_factor = 1 / pb_ratio.replace(0, np.nan)
    value_factor = value_factor.replace([np.inf, -np.inf], 0)
    return value_factor


def main() -> None:
    parser = argparse.ArgumentParser(description="Fama-French 因子计算")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None, help="起始日期")
    parser.add_argument("--end", default=None, help="结束日期")
    parser.add_argument("--names", nargs="*",
                        help="指定因子: ff_mkt ff_smb ff_hml ff_size ff_value (默认全部)")
    parser.add_argument("--symbols", nargs="*", help="指定股票列表(默认全市场)")
    parser.add_argument("--skip-write", action="store_true", help="仅计算不写入数据库")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = args.start or cfg["backtest"]["start_date"]
    end = args.end or cfg["backtest"]["end_date"]

    db = Database(get_db_path(cfg))

    print(f"Fama-French 因子计算: {start} ~ {end}")

    bars = db.get_daily_bars(symbols=args.symbols, start_date=start, end_date=end)
    if bars.empty:
        print("无日线数据")
        db.close()
        return

    symbols = bars["symbol"].unique().tolist()
    print(f"股票数量: {len(symbols)}")

    bars_pivot = bars.pivot(index="date", columns="symbol", values="close")
    returns = bars.pivot(index="date", columns="symbol", values="close").pct_change()

    names = args.names or ["ff_mkt", "ff_smb", "ff_hml", "ff_size", "ff_value"]

    if "ff_smb" in names or "ff_hml" in names or "ff_size" in names or "ff_value" in names:
        print("从 AKShare 获取市值和 PB 数据...")
        akshare_data = _get_akshare_data(symbols, start, end)
        market_cap = akshare_data["market_cap"]
        pb_ratio = akshare_data["pb"]
        print(f"市值数据: {market_cap.shape if not market_cap.empty else '空'}")
        print(f"PB数据: {pb_ratio.shape if not pb_ratio.empty else '空'}")
    else:
        market_cap = pd.DataFrame()
        pb_ratio = pd.DataFrame()

    results = {}

    if "ff_mkt" in names:
        print("计算 MKT 因子...")
        results["ff_mkt"] = compute_mkt(bars_pivot)

    if "ff_smb" in names:
        print("计算 SMB 因子...")
        results["ff_smb"] = compute_smb(bars_pivot, market_cap, returns)

    if "ff_hml" in names:
        print("计算 HML 因子...")
        results["ff_hml"] = compute_hml(bars_pivot, pb_ratio, returns)

    if "ff_size" in names:
        print("计算 Size 横截面因子...")
        results["ff_size"] = compute_size_cross_section(bars_pivot, market_cap)

    if "ff_value" in names:
        print("计算 Value 横截面因子...")
        results["ff_value"] = compute_value_cross_section(bars_pivot, pb_ratio)

    if args.skip_write:
        print("跳过写入数据库")
        for name, df in results.items():
            if not df.empty:
                print(f"  {name}: {df.shape}")
        db.close()
        return

    print("写入数据库...")
    for name, data in results.items():
        if data.empty:
            continue

        if isinstance(data, pd.Series):
            df = data.reset_index()
            df.columns = ["date", name]
        else:
            df = data.reset_index().melt(id_vars="date", var_name="symbol", value_name=name)

        df = df.dropna()
        if df.empty:
            continue

        try:
            db.save_factors(df[["date", "symbol", name]])
            print(f"  {name}: {len(df)} 条")
        except Exception as e:
            print(f"  {name} 写入失败: {e}")

    print("完成!")
    db.close()


if __name__ == "__main__":
    main()
