import pandas as pd
import numpy as np
from core.factors.base.factor_hub import FactorHub, register_factor, FactorContext, list_all


def test_factor_hub_empty():
    hub = FactorHub()
    cats = hub.categories()
    assert isinstance(cats, list)
    assert len(cats) == 0


def test_factor_hub_register_and_get():
    hub = FactorHub()

    @hub.register(name="test_momentum", category="test")
    def momentum(ctx: FactorContext):
        return pd.DataFrame()

    factors = hub.list_by_category("test")
    assert "test_momentum" in factors

    meta = hub.get("test_momentum")
    assert meta.func is momentum
    assert meta.category == "test"


def test_factor_hub_categories():
    hub = FactorHub()

    @hub.register(name="ma_5", category="tech")
    def ma5(ctx):
        return pd.DataFrame()

    tech_factors = hub.list_by_category("tech")
    assert "ma_5" in tech_factors


def test_factor_context_named_fields():
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = FactorContext(close=df)
    assert ctx.close is not None
    assert ctx.open is None
    assert ctx.high is None


def test_factor_hub_list_all():
    hub = FactorHub()

    @hub.register(name="f1", category="a")
    def f1(ctx):
        return pd.DataFrame()

    @hub.register(name="f2", category="b")
    def f2(ctx):
        return pd.DataFrame()

    all_factors = hub.list_all()
    assert "f1" in all_factors
    assert "f2" in all_factors


def test_register_factor_decorator():
    @register_factor(name="decorator_test", category="test_cat")
    def my_factor(ctx: FactorContext):
        return pd.DataFrame()

    all_f = list_all()
    assert "decorator_test" in all_f


def test_factor_hub_multi_categories():
    hub = FactorHub()

    @hub.register(name="f1", category="cat1")
    def f1(ctx):
        return pd.DataFrame()

    @hub.register(name="f2", category="cat2")
    def f2(ctx):
        return pd.DataFrame()

    cats = hub.categories()
    assert "cat1" in cats
    assert "cat2" in cats


def test_factor_compute_with_realistic_data():
    hub = FactorHub()

    @hub.register(name="realistic_factor", category="realistic",
                  requires=["close"])
    def realistic(ctx: FactorContext):
        close = ctx.close
        return close.rank(pct=True)

    bars = pd.DataFrame({
        "symbol": ["A", "B", "C"] * 2,
        "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
        "close": [10.0, 20.0, 30.0, 11.0, 22.0, 33.0],
    })

    result = hub.compute_all(bars, names=["realistic_factor"])
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "realistic_factor" in result["factor_name"].values
