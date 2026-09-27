import numpy as np
import pandas as pd
import pytest

from credit_risk.scorecard import Scorecard, points_params


def _data(n=30000, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "noise": rng.normal(size=n),
                      "cat": rng.choice(["a", "b"], n)})
    logit = -2 + 0.8 * X.x1 - 0.6 * X.x2 + np.where(X.cat == "b", 0.5, 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_selects_informative_and_negative_coefficients():
    X, y = _data()
    sc = Scorecard.fit(X, y, ["x1", "x2", "noise"], ["cat"])
    assert {"x1", "x2", "cat"} <= set(sc.features_) and "noise" not in sc.features_
    assert (sc.coefficients() < 0).all()        # high WoE = low risk -> negative coefficient


def test_points_scale():
    factor, offset = points_params()
    assert factor == pytest.approx(20 / np.log(2))
    assert offset + factor * np.log(50) == pytest.approx(600)


def test_score_decreases_with_pd():
    X, y = _data()
    sc = Scorecard.fit(X, y, ["x1", "x2"], ["cat"])
    pd_, s = sc.predict_pd(X), sc.score(X)
    assert ((pd_ > 0) & (pd_ < 1)).all()
    assert np.corrcoef(pd_, s)[0, 1] < -0.9
