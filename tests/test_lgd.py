import numpy as np
import pandas as pd

from credit_risk.lgd import TwoStageLGD, expected_loss, exposure_at_default, realized_lgd


def test_realized_lgd_bounds():
    df = pd.DataFrame({"funded_amnt": [1000, 1000, 1000, 1000, 1000],
                       "total_rec_prncp": [400, 400, 400, 1000, 400],
                       "recoveries": [0, 600, 300, 50, 900],
                       "collection_recovery_fee": [0, 0, 0, 0, 0]})
    lgd = realized_lgd(df)
    assert exposure_at_default(df).tolist() == [600, 600, 600, 0, 600]
    assert lgd[0] == 1.0 and lgd[1] == 0.0 and lgd[2] == 0.5
    assert np.isnan(lgd[3])                     # EAD = 0
    assert lgd[4] == 0.0                        # recupera más que la exposición -> recortado a 0


def test_two_stage_predictions_in_range():
    rng = np.random.default_rng(4); n = 5000
    X = pd.DataFrame({"x": rng.normal(size=n), "cat": rng.choice(["a", "b"], n)})
    rec = rng.random(n) < 1 / (1 + np.exp(-X.x))
    lgd = np.where(rec, np.clip(0.6 - 0.1 * X.x + rng.normal(0, 0.1, n), 0, 0.99), 1.0)
    m = TwoStageLGD(["x"], ["cat"]).fit(X, lgd)
    p = m.predict(X)
    assert ((p >= 0) & (p <= 1)).all() and abs(p.mean() - lgd.mean()) < 0.05


def test_expected_loss_product():
    assert expected_loss(np.array([0.1]), np.array([0.5]), np.array([1000.0]))[0] == 50.0
