from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk import config
from credit_risk.woe import WoeBinner


def points_params(pdo=config.PDO, base_score=config.BASE_SCORE, base_odds=config.BASE_ODDS):
    """Points scaling: base_score points at base_odds (good:bad), +pdo points each time the odds double."""
    factor = pdo / np.log(2)
    return factor, base_score - factor * np.log(base_odds)


@dataclass
class Scorecard:
    binner: WoeBinner
    features_: list[str]
    model_: LogisticRegression

    @classmethod
    def fit(cls, X, y, numeric, categorical, iv_min=0.02, corr_max=0.7) -> "Scorecard":
        binner = WoeBinner(numeric, categorical).fit(X, y)
        W = binner.transform(X)
        selected: list[str] = []
        for c in binner.iv_[binner.iv_ >= iv_min].sort_values(ascending=False).index:
            if all(abs(W[c].corr(W[s])) <= corr_max for s in selected):
                selected.append(c)
        model = LogisticRegression(C=np.inf, max_iter=2000).fit(W[selected], y)   # unpenalized
        return cls(binner, selected, model)

    def predict_pd(self, X) -> np.ndarray:
        return self.model_.predict_proba(self.binner.transform(X)[self.features_])[:, 1]

    def score(self, X) -> np.ndarray:
        factor, offset = points_params()
        p = np.clip(self.predict_pd(X), 1e-9, 1 - 1e-9)
        return offset + factor * np.log((1 - p) / p)

    def coefficients(self) -> pd.Series:
        return pd.Series(self.model_.coef_[0], index=self.features_)
