import numpy as np
import pandas as pd

from credit_risk.woe import WoeBinner


def _data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n); noise = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    p = 0.05 + 0.4 * x + np.where(cat == "c", 0.2, 0)
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"x": x, "noise": noise, "cat": cat}), y


def test_iv_ranks_informative_over_noise():
    X, y = _data()
    b = WoeBinner(["x", "noise"], ["cat"]).fit(X, y)
    assert b.iv_["x"] > 0.3 and b.iv_["noise"] < 0.02 and b.iv_["cat"] > 0.02


def test_woe_monotone_for_monotone_risk():
    X, y = _data()
    t = WoeBinner(["x"], []).fit(X, y).woe_table("x")
    t = t[t["bin"] != "MISSING"]
    assert (np.diff(t["woe"].to_numpy()) < 0).all()      # más x -> más riesgo -> menor WoE


def test_transform_unseen_and_missing():
    X, y = _data()
    b = WoeBinner(["x"], ["cat"]).fit(X, y)
    new = pd.DataFrame({"x": [np.nan, 0.5], "cat": ["zzz", None]})
    W = b.transform(new)
    assert not W.isna().any().any()
    assert W.loc[0, "cat"] == b.woe_table("cat").set_index("bin").get("woe").get("OTHER", 0.0)


def test_woe_finite_with_pure_bin():
    X = pd.DataFrame({"c": ["pure"] * 50 + ["mix"] * 50})
    y = np.r_[np.zeros(50), np.r_[np.zeros(25), np.ones(25)]].astype(int)
    b = WoeBinner([], ["c"], min_share=0.0).fit(X, y)
    assert np.isfinite(b.woe_table("c")["woe"]).all()
