from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WoeBinner:
    """Weight-of-evidence binning. Sign convention: WoE = ln(%goods / %bads), so high WoE means low risk."""

    numeric: list[str]
    categorical: list[str]
    n_bins: int = 10
    min_share: float = 0.01
    smoothing: float = 0.5
    edges_: dict = field(default_factory=dict)
    levels_: dict = field(default_factory=dict)
    tables_: dict = field(default_factory=dict)
    iv_: pd.Series = None

    def fit(self, X: pd.DataFrame, y) -> "WoeBinner":
        y = np.asarray(y)
        for c in self.numeric:
            edges = np.unique(np.nanquantile(X[c].astype(float), np.linspace(0, 1, self.n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges_[c] = edges
        for c in self.categorical:
            share = X[c].value_counts(normalize=True)
            self.levels_[c] = set(share[share >= self.min_share].index)
        B = self.bins(X)
        n_bad, n_good = y.sum(), (1 - y).sum()
        s = self.smoothing
        for c in B.columns:
            t = pd.DataFrame({"bin": B[c], "y": y}).groupby("bin", sort=False)["y"].agg(["size", "sum"])
            t = t.rename(columns={"size": "n", "sum": "bads"})
            t["goods"] = t["n"] - t["bads"]
            dg = (t["goods"] + s) / (n_good + s)
            db = (t["bads"] + s) / (n_bad + s)
            t["woe"] = np.log(dg / db)
            t["iv"] = (dg - db) * t["woe"]
            self.tables_[c] = self._order(c, t.reset_index())
        self.iv_ = pd.Series({c: float(t["iv"].sum()) for c, t in self.tables_.items()})
        return self

    def _order(self, c: str, t: pd.DataFrame) -> pd.DataFrame:
        if c in self.edges_:
            def key(b):
                return np.inf if b == "MISSING" else float(b.split(",")[0].strip("(["))
            t = t.assign(_k=t["bin"].map(key)).sort_values("_k").drop(columns="_k")
        return t[["bin", "n", "bads", "goods", "woe", "iv"]].reset_index(drop=True)

    def bins(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.numeric:
            b = pd.cut(X[c].astype(float), self.edges_[c], right=True).astype(str)
            out[c] = b.where(X[c].notna(), "MISSING")
        for c in self.categorical:
            v = X[c].where(X[c].isin(self.levels_[c]), "OTHER")
            out[c] = v.where(X[c].notna(), "MISSING").astype(str)
        return pd.DataFrame(out, index=X.index)

    def woe_table(self, col: str) -> pd.DataFrame:
        return self.tables_[col].copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        B = self.bins(X)
        return pd.DataFrame({c: B[c].map(self.tables_[c].set_index("bin")["woe"]).fillna(0.0).astype(float)
                             for c in B.columns}, index=X.index)
