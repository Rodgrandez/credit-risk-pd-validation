import pandas as pd

from credit_risk import config


def split_samples(df: pd.DataFrame, holdout_frac: float = config.HOLDOUT_FRAC) -> dict[str, pd.DataFrame]:
    dev = df[df["vintage"].isin(config.TRAIN_YEARS)]
    oot = df[df["vintage"].isin(config.OOT_YEARS)]
    bucket = pd.util.hash_pandas_object(dev["id"].astype(str), index=False).to_numpy() % 10_000
    is_holdout = bucket < holdout_frac * 10_000
    return {"train": dev[~is_holdout].reset_index(drop=True),
            "holdout": dev[is_holdout].reset_index(drop=True),
            "oot": oot.reset_index(drop=True)}


def early_stopping_split(train: pd.DataFrame, frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve an early-stopping set out of train (salted hash), so the holdout stays untouched by tuning."""
    key = "es-" + train["id"].astype(str)
    bucket = pd.util.hash_pandas_object(key, index=False).to_numpy() % 10_000
    is_es = bucket < frac * 10_000
    return train[~is_es].reset_index(drop=True), train[is_es].reset_index(drop=True)
