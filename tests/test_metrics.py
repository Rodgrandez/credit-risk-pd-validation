import numpy as np
import pandas as pd
import pytest

from credit_risk.metrics import (auc, binomial_upper_pvalue, calibration_table, gini, hosmer_lemeshow, ks,
                                 psi, psi_categorical)


def test_perfect_and_random_discrimination():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, p) == 1.0 and gini(y, p) == 1.0 and ks(y, p) == 1.0
    rng = np.random.default_rng(0); y2 = rng.integers(0, 2, 20000); p2 = rng.random(20000)
    assert abs(auc(y2, p2) - 0.5) < 0.02


def test_psi_identical_zero_and_shift_large():
    rng = np.random.default_rng(1); e = rng.normal(0, 1, 50000)
    assert psi(e, rng.normal(0, 1, 50000)) < 0.01
    assert psi(e, rng.normal(1.5, 1, 50000)) > 0.25


def test_psi_handles_empty_bin():
    e = np.r_[np.zeros(100), np.arange(100.0)]
    a = np.arange(100.0)                      # sin masa en el bin del 0 repetido
    val = psi(e, a)
    assert np.isfinite(val) and val > 0


def test_psi_categorical():
    e = pd.Series(["a"] * 50 + ["b"] * 50); a = pd.Series(["a"] * 50 + ["c"] * 50)
    assert psi_categorical(e, e) == pytest.approx(0, abs=1e-9)
    assert psi_categorical(e, a) > 0.25


def test_hosmer_lemeshow_calibrated_vs_miscalibrated():
    rng = np.random.default_rng(2); p = rng.uniform(0.02, 0.3, 40000); y = rng.random(40000) < p
    assert hosmer_lemeshow(y.astype(int), p)[1] > 0.01
    assert hosmer_lemeshow(y.astype(int), np.clip(p * 2, 0, 0.99))[1] < 1e-6


def test_binomial_upper_pvalue():
    assert binomial_upper_pvalue(50, 100, 0.1) < 1e-10
    assert binomial_upper_pvalue(10, 100, 0.1) > 0.3


def test_calibration_table_columns():
    y = np.array([0, 1, 0, 0, 1, 1]); p = np.array([.1, .9, .2, .1, .8, .7]); g = ["a", "a", "a", "b", "b", "b"]
    t = calibration_table(y, p, g)
    assert list(t.columns) == ["group", "n", "defaults", "observed_dr", "mean_pd", "binom_pvalue"]
    assert t.loc[t.group == "a", "defaults"].item() == 1
