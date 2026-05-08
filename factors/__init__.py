"""因子库入口。

`import factors` 会触发以下因子集合向 FactorHub 注册:
  - alpha101_full: WorldQuant Alpha101 全 101 个公式因子
  - gtja191_full:  国泰君安 GTJA 191 因子 (1-70)
  - technical:    传统技术因子(动量/RSI/MACD/布林/量比/波动率,共 13 个)

新增因子文件时,只需在此处加一行 `from . import xxx`,无需改动调用方。
"""
# 静音因子计算时的已知非致命告警:
#   - pandas FutureWarning: pct_change 默认 fill_method 即将变更(我们已验证行为)
#   - numpy  RuntimeWarning: 全 NaN 列上的 argmax/argmin reduce(长期停牌股票,返回 NaN 合理)
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*fill_method='pad' in DataFrame\.pct_change.*",
    category=FutureWarning,
)
_warnings.filterwarnings("ignore", message=r".*invalid value encountered in reduce.*", category=RuntimeWarning)
_warnings.filterwarnings("ignore", message=r".*All-NaN slice encountered.*", category=RuntimeWarning)

from . import alpha101_full  # noqa: F401
from . import gtja191_full   # noqa: F401
from . import technical      # noqa: F401

__all__ = ["alpha101_full", "gtja191_full", "technical"]