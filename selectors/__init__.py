"""DEPRECATED: 此包名与 Python 标准库 `selectors` 冲突。

当 Python 启动期(numpy/subprocess 内部)`import selectors` 被解析到本目录时,
我们必须立刻把自己替换为标准库的同名模块,否则后续 `selectors.SelectSelector` 等
访问会全部失败。

新代码请直接 `from stock_selectors.factor_rank import FactorRankSelector`。
"""
import sys as _sys
import importlib.util as _ilu
import os as _os

# 找到标准库 selectors 模块的真实路径(绕开 sys.path,直接从 stdlib 目录加载)
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
_stdlib_path = None
for _p in _sys.path:
    if not _p or _os.path.abspath(_p) == _this_dir:
        continue
    _candidate = _os.path.join(_p, "selectors.py")
    if _os.path.isfile(_candidate):
        _stdlib_path = _candidate
        break

if _stdlib_path:
    _spec = _ilu.spec_from_file_location("selectors", _stdlib_path)
    _stdlib_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_stdlib_mod)
    # 用标准库模块替换本模块
    _sys.modules["selectors"] = _stdlib_mod
    # 把标准库的属性导出到本命名空间(以防当前 frame 已绑定)
    for _k in dir(_stdlib_mod):
        if not _k.startswith("_"):
            globals()[_k] = getattr(_stdlib_mod, _k)
else:
    # 找不到标准库,做最后的兜底,避免后续 import 报 AttributeError
    raise ImportError(
        "stdlib selectors not found; please import from stock_selectors instead"
    )