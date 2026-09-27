import numpy as np

from credit_risk import config
from credit_risk.data import clean, feature_frame, parse_emp_length


def test_raw_columns_cover_features():
    needed = {"fico_range_low", "fico_range_high", "earliest_cr_line", "emp_length", "issue_d", "loan_status"}
    assert needed <= set(config.RAW_COLUMNS)


def test_parse_emp_length():
    assert parse_emp_length("10+ years") == 10
    assert parse_emp_length("< 1 year") == 0
    assert parse_emp_length("3 years") == 3
    assert parse_emp_length("1 year") == 1
    assert np.isnan(parse_emp_length(None))


def test_clean_filters_and_target(raw_frame):
    df = clean(raw_frame)
    assert df["id"].str.isnumeric().all()                      # summary (junk) rows dropped
    assert set(df["default"].unique()) <= {0, 1}
    assert not df["vintage"].isna().any()
    kept = raw_frame.set_index("id").loc[df["id"]]
    assert (kept["term"].str.strip() == "36 months").all()      # 36-month loans only
    assert not kept["loan_status"].isin(["Current"]).any()      # resolved loans only
    bad = kept["loan_status"].isin(config.BAD_STATUSES).astype(int).to_numpy()
    assert (df["default"].to_numpy() == bad).all()


def test_clean_derived_features(raw_frame):
    df = clean(raw_frame)
    raw_low = raw_frame.set_index("id").loc[df["id"], "fico_range_low"].to_numpy()
    assert np.allclose(df["fico_mid"], raw_low + 2)
    hist = df["credit_history_years"].dropna()
    assert (hist > 0).all()


def test_feature_frame_excludes_leakage(raw_frame):
    X = feature_frame(clean(raw_frame))
    forbidden = config.LEAKAGE_COLUMNS | {"grade", "sub_grade", "int_rate", "installment", "default"}
    assert not (set(X.columns) & forbidden)
    assert list(X.columns) == config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
