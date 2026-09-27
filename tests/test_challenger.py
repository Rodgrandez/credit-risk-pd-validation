import numpy as np
import pandas as pd

from credit_risk.challenger import fit_xgb, shap_matrix, to_model_frame
from credit_risk.metrics import auc


def _data(n=8000, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)})
    logit = -1.5 + 1.2 * X.x + np.where(X.cat == "c", 1.0, 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_xgb_learns_signal():
    X, y = _data()
    tr, cats = to_model_frame(X.iloc[:6000], ["cat"])
    va, _ = to_model_frame(X.iloc[6000:], ["cat"], cats)
    m = fit_xgb(tr, y[:6000], va, y[6000:])
    p = m.predict_proba(va)[:, 1]
    assert ((p >= 0) & (p <= 1)).all() and auc(y[6000:], p) > 0.7


def test_unseen_category_in_valid():
    X, y = _data()
    tr, cats = to_model_frame(X, ["cat"])
    new, _ = to_model_frame(pd.DataFrame({"x": [0.1], "cat": ["zzz"]}), ["cat"], cats)
    m = fit_xgb(tr, y, tr, y)
    assert np.isfinite(m.predict_proba(new)[:, 1]).all()


def test_shap_shape():
    X, y = _data()
    tr, _ = to_model_frame(X, ["cat"])
    m = fit_xgb(tr, y, tr, y)
    vals, sample = shap_matrix(m, tr, max_rows=500)
    assert vals.shape == sample.shape == (500, 2)
