from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from credit_risk.woe import WoeBinner


def exposure_at_default(df: pd.DataFrame) -> pd.Series:
    return df["funded_amnt"] - df["total_rec_prncp"]


def realized_lgd(df: pd.DataFrame) -> pd.Series:
    """LGD = 1 - net recoveries / EAD, clipped to [0, 1]; NaN when EAD <= 0."""
    ead = exposure_at_default(df)
    net = df["recoveries"] - df["collection_recovery_fee"]
    lgd = (1 - net / ead.where(ead > 0)).clip(0, 1)
    return lgd.where(ead > 0)


@dataclass
class TwoStageLGD:
    """Stage 1: probability of any recovery. Stage 2: LGD given some recovery. No recovery means LGD = 1."""

    numeric: list[str]
    categorical: list[str]
    binner_: WoeBinner = None
    stage1_: LogisticRegression = field(default=None)
    stage2_: LinearRegression = field(default=None)

    def fit(self, X: pd.DataFrame, lgd) -> "TwoStageLGD":
        lgd = np.asarray(lgd, dtype=float)
        recovered = (lgd < 1).astype(int)
        self.binner_ = WoeBinner(self.numeric, self.categorical).fit(X, recovered)
        W = self.binner_.transform(X)
        self.stage1_ = LogisticRegression(max_iter=2000).fit(W, recovered)
        self.stage2_ = LinearRegression().fit(W[recovered == 1], lgd[recovered == 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        W = self.binner_.transform(X)
        p_rec = self.stage1_.predict_proba(W)[:, 1]
        lgd_rec = np.clip(self.stage2_.predict(W), 0, 1)
        return p_rec * lgd_rec + (1 - p_rec) * 1.0


def expected_loss(pd_, lgd_, ead) -> np.ndarray:
    return np.asarray(pd_) * np.asarray(lgd_) * np.asarray(ead, dtype=float)


def ead_factor(defaults: pd.DataFrame) -> float:
    """Average exposure at default as a share of the funded amount, estimated on development defaults."""
    ead, funded = exposure_at_default(defaults), defaults["funded_amnt"]
    ok = (ead > 0) & (funded > 0)
    return float((ead[ok] / funded[ok]).mean())
