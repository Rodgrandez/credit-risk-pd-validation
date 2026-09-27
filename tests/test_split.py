import pandas as pd

from credit_risk import config
from credit_risk.data import clean
from credit_risk.split import split_samples


def test_split_disjoint_and_by_vintage(raw_frame):
    df = clean(raw_frame)
    s = split_samples(df)
    assert set(s) == {"train", "holdout", "oot"}
    assert s["train"]["vintage"].isin(config.TRAIN_YEARS).all()
    assert s["holdout"]["vintage"].isin(config.TRAIN_YEARS).all()
    assert s["oot"]["vintage"].isin(config.OOT_YEARS).all()
    ids = [set(v["id"]) for v in s.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_deterministic_and_order_invariant(raw_frame):
    df = clean(raw_frame)
    a = split_samples(df)["holdout"]["id"]
    b = split_samples(df.sample(frac=1, random_state=3))["holdout"]["id"]
    assert set(a) == set(b)


def test_holdout_fraction_close():
    df = pd.DataFrame({"id": [str(i) for i in range(1, 20001)], "vintage": 2012})
    s = split_samples(df)
    assert abs(len(s["holdout"]) / len(df) - config.HOLDOUT_FRAC) < 0.02
