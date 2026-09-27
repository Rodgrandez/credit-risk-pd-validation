import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from credit_risk import config


def to_model_frame(X: pd.DataFrame, categorical: list[str], categories: dict | None = None):
    """Cast categoricals to a fixed category set (from training); unseen levels become missing."""
    X = X.copy()
    categories = categories or {c: sorted(X[c].dropna().astype(str).unique()) for c in categorical}
    for c in categorical:
        X[c] = pd.Categorical(X[c].astype("string"), categories=categories[c])
    return X, categories


def fit_xgb(X_train, y_train, X_valid, y_valid, seed: int = config.SEED) -> XGBClassifier:
    model = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=50, eval_metric="auc",
                          early_stopping_rounds=50, tree_method="hist", enable_categorical=True,
                          random_state=seed)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def shap_matrix(model: XGBClassifier, X: pd.DataFrame, max_rows: int = 5000, seed: int = config.SEED):
    sample = X.sample(n=min(max_rows, len(X)), random_state=seed)
    values = shap.TreeExplainer(model).shap_values(sample)
    return np.asarray(values), sample
