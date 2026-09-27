import numpy as np
import pandas as pd

from credit_risk.lgd import expected_loss, exposure_at_default
from credit_risk.metrics import auc, calibration_table, gini, ks, psi, psi_categorical


def discrimination_table(samples: dict) -> pd.DataFrame:
    rows = [{"sample": k, "n": len(y), "default_rate": float(np.mean(y)), "auc": auc(y, p),
             "gini": gini(y, p), "ks": ks(y, p)} for k, (y, p) in samples.items()]
    return pd.DataFrame(rows)


def decile_calibration(y, pd_) -> pd.DataFrame:
    deciles = pd.qcut(pd.Series(pd_).rank(method="first"), 10, labels=False) + 1
    return calibration_table(y, pd_, deciles)


def _status(v: float) -> str:
    return "stable" if v < 0.10 else ("monitor" if v < 0.25 else "shift")


def stability_table(X_ref, X_new, numeric, categorical, score_ref, score_new) -> pd.DataFrame:
    rows = [{"variable": c, "psi": psi(X_ref[c], X_new[c])} for c in numeric]
    rows += [{"variable": c, "psi": psi_categorical(X_ref[c], X_new[c])} for c in categorical]
    rows.append({"variable": "score", "psi": psi(score_ref, score_new)})
    t = pd.DataFrame(rows)
    t["status"] = t["psi"].map(_status)
    return t


def el_backtest(df: pd.DataFrame, pd_, lgd_pred, ead_factor: float = 1.0) -> pd.DataFrame:
    """Lifetime expected loss (PD x LGD x EAD) vs realized loss (EAD at default - net recoveries), by vintage.

    LGD is defined on exposure at default, so the predicted EAD is funded amount x ead_factor (the average
    EAD/funded ratio of development defaults): both sides of the ratio use the same exposure base.
    """
    net_recovery = df["recoveries"] - df["collection_recovery_fee"]
    d = df.assign(el=expected_loss(pd_, lgd_pred, df["funded_amnt"] * ead_factor),
                  loss=df["default"] * (exposure_at_default(df) - net_recovery).clip(lower=0))
    t = d.groupby("vintage").agg(n=("el", "size"), exposure=("funded_amnt", "sum"), expected_loss=("el", "sum"),
                                 realized_loss=("loss", "sum")).reset_index()
    t["ratio"] = t["expected_loss"] / t["realized_loss"]
    return t


def sensitivity_table(df: pd.DataFrame, pd_, lgd_pred, ead_factor: float = 1.0) -> pd.DataFrame:
    pd_, lgd_pred, ead = np.asarray(pd_), np.asarray(lgd_pred), df["funded_amnt"].to_numpy() * ead_factor
    scen = {"base": (pd_, lgd_pred), "PD x1.2": (np.clip(pd_ * 1.2, 0, 1), lgd_pred),
            "LGD +10pp": (pd_, np.clip(lgd_pred + 0.10, 0, 1)),
            "PD x1.2 & LGD +10pp": (np.clip(pd_ * 1.2, 0, 1), np.clip(lgd_pred + 0.10, 0, 1))}
    return pd.DataFrame([{"scenario": k, "expected_loss": float(expected_loss(p, lg, ead).sum())}
                         for k, (p, lg) in scen.items()])
