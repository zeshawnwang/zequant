"""
V1 组件化管道 — 老文件保留为快捷入口。

实际实现在: core/strategies/impl/v1_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.strategies.impl.v1_pipeline import run_v1_pipeline

if __name__ == "__main__":
    run_v1_pipeline()
