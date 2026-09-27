import numpy as np
import pandas as pd
import pytest

from credit_risk.validation import (decile_calibration, discrimination_table, el_backtest, sensitivity_table,
                                    stability_table)


def test_discrimination_table():
    y = np.array([0, 1, 0, 1]); p = np.array([.1, .9, .2, .8])
    t = discrimination_table({"train": (y, p), "oot": (y, p[::-1])})
    assert list(t.columns) == ["sample", "n", "default_rate", "auc", "gini", "ks"]
    assert t.set_index("sample").loc["train", "auc"] == 1.0


def test_decile_calibration_ten_groups():
    rng = np.random.default_rng(0); p = rng.uniform(0, .3, 5000); y = (rng.random(5000) < p).astype(int)
    t = decile_calibration(y, p)
    assert t["group"].tolist() == list(range(1, 11)) and t["n"].sum() == 5000


def test_stability_status():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(size=5000), "c": rng.choice(["x", "y"], 5000)})
    new = pd.DataFrame({"a": rng.normal(2, 1, 5000), "c": rng.choice(["x", "y"], 5000)})
    t = stability_table(ref, new, ["a"], ["c"], ref["a"].to_numpy(), ref["a"].to_numpy()).set_index("variable")
    assert t.loc["a", "status"] == "shift" and t.loc["c", "status"] == "stable" and t.loc["score", "psi"] < 0.01


def _portfolio():
    return pd.DataFrame({"vintage": [2014, 2014, 2015], "default": [1, 0, 1], "funded_amnt": [1000., 1000., 2000.],
                         "total_rec_prncp": [400., 1000., 0.], "recoveries": [100., 0., 0.],
                         "collection_recovery_fee": [0., 0., 0.]})


def test_el_backtest():
    df = _portfolio()
    t = el_backtest(df, np.array([.1, .1, .2]), np.array([.5, .5, .5])).set_index("vintage")
    assert t.loc[2014, "expected_loss"] == 100.0          # .1*.5*1000 *2
    assert t.loc[2014, "realized_loss"] == 500.0          # (600-100)
    assert t.loc[2015, "realized_loss"] == 2000.0


def test_sensitivity():
    df = _portfolio()
    t = sensitivity_table(df, np.array([.1, .1, .2]), np.array([.5, .5, .5])).set_index("scenario")
    assert t.loc["PD x1.2", "expected_loss"] == pytest.approx(1.2 * t.loc["base", "expected_loss"])
    assert t.loc["LGD +10pp", "expected_loss"] > t.loc["base", "expected_loss"]
