"""mss_dynamic — 实盘信号生成入口。

拆分子模块:
  mss_state   状态持久化
  mss_factors 因子与择时信号计算  
  mss_signal  订单生成与主流程
  mss_report  HTML 报告与邮件发送
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .mss_signal import main

if __name__ == "__main__":
    main()
