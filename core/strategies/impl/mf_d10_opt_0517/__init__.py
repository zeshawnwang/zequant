"""MF_D10_OPT_0517 — 策略组装入口。

从 build.py 导入构建函数，供 strategies.hub 注册使用。
"""
from .build import build_mf_d10_opt_0517

__all__ = ["build_mf_d10_opt_0517"]
