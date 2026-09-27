import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

_FLOOR = 1e-6


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def _shares(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    nan = np.isnan(x)
    counts = np.histogram(x[~nan], bins=edges)[0]
    return np.clip(np.append(counts, nan.sum()) / len(x), _FLOOR, None)


def psi(expected, actual, n_bins: int = 10) -> float:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe, pa = _shares(e, edges), _shares(a, edges)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorical(expected, actual) -> float:
    e = pd.Series(expected).fillna("MISSING").value_counts(normalize=True)
    a = pd.Series(actual).fillna("MISSING").value_counts(normalize=True)
    idx = e.index.union(a.index)
    pe = e.reindex(idx, fill_value=0).clip(lower=_FLOOR)
    pa = a.reindex(idx, fill_value=0).clip(lower=_FLOOR)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def hosmer_lemeshow(y, p, n_groups: int = 10, df: int | None = None) -> tuple[float, float]:
    """HL test. df defaults to n_groups - 2 (development sample); use df=n_groups on external / OOT data."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["g"] = pd.qcut(d["p"].rank(method="first"), n_groups, labels=False)
    g = d.groupby("g").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    stat = float((((g.obs - g.exp) ** 2) / (g.exp * (1 - g.exp / g.n))).sum())
    return stat, float(chi2.sf(stat, n_groups - 2 if df is None else df))


def calibration_summary(y, p) -> dict:
    """Effect sizes to read alongside HL: calibration-in-the-large, slope, Brier and the largest decile gap."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = float(LogisticRegression(C=np.inf, max_iter=1000).fit(logit, y).coef_[0][0])
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    gaps = pd.DataFrame({"y": y, "p": p, "d": dec}).groupby("d").mean()
    return {"mean_pd": float(p.mean()), "observed_dr": float(y.mean()), "ratio": float(p.mean() / y.mean()),
            "slope": slope, "brier": float(np.mean((p - y) ** 2)),
            "max_decile_gap_pp": float(100 * (gaps["y"] - gaps["p"]).abs().max())}


def format_pvalue(p: float) -> str:
    return "<1e-16" if p < 1e-16 else f"{p:.3g}"


def binomial_upper_pvalue(defaults: int, n: int, pd_mean: float) -> float:
    return float(binomtest(int(defaults), int(n), float(pd_mean), alternative="greater").pvalue)


def calibration_table(y, p, groups) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "group": np.asarray(groups)})
    t = d.groupby("group").agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean")).reset_index()
    t["observed_dr"] = t["defaults"] / t["n"]
    t["binom_pvalue"] = [binomial_upper_pvalue(k, n, m) for k, n, m in zip(t.defaults, t.n, t.mean_pd)]
    return t[["group", "n", "defaults", "observed_dr", "mean_pd", "binom_pvalue"]]
