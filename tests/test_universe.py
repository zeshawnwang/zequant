"""Universe 涨跌停板块分级判定单元测试。"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.universe import get_price_limit_pct, UniverseConfig


def test_main_board_10pct():
    """沪深主板:6/0 开头(非创业板/科创板)±10%。"""
    assert get_price_limit_pct("600000") == 10.0
    assert get_price_limit_pct("000001") == 10.0
    assert get_price_limit_pct("002001") == 10.0


def test_chinext_20pct():
    """创业板 30/301 开头:±20%。"""
    assert get_price_limit_pct("300001") == 20.0
    assert get_price_limit_pct("301010") == 20.0


def test_star_market_20pct():
    """科创板 688/689 开头:±20%。"""
    assert get_price_limit_pct("688001") == 20.0
    assert get_price_limit_pct("689001") == 20.0


def test_bj_30pct():
    """北交所 4/8 开头:±30%。"""
    assert get_price_limit_pct("830001") == 30.0
    assert get_price_limit_pct("872001") == 30.0


def test_st_5pct():
    """ST 股(无论板块):±5%。"""
    assert get_price_limit_pct("000001", is_st=True) == 5.0
    assert get_price_limit_pct("300001", is_st=True) == 5.0
    assert get_price_limit_pct("688001", is_st=True) == 5.0


def test_universe_config_repr():
    """UniverseConfig 应当有清晰的 __repr__。"""
    cfg = UniverseConfig.from_config({
        "exclude": ["ST股", {"上市不满N天": 60}],
        "min_daily_amount": 100_000_000,
    })
    s = repr(cfg)
    assert "UniverseConfig" in s
    assert "60" in s
    assert "100,000,000" in s


def test_universe_config_default():
    """空配置应当能默认构造。"""
    cfg = UniverseConfig.from_config({})
    assert cfg.exclude_st is True
    assert cfg.min_listed_days == 60


if __name__ == "__main__":
    test_main_board_10pct()
    test_chinext_20pct()
    test_star_market_20pct()
    test_bj_30pct()
    test_st_5pct()
    test_universe_config_repr()
    test_universe_config_default()
    print("Universe 板块分级判定测试通过")