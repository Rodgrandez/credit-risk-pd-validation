import numpy as np
import pandas as pd
import pytest


def make_raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    years = rng.choice([2010, 2012, 2014, 2015, 2016], size=n)
    months = rng.choice(["Jan", "Jun", "Dec"], size=n)
    status = rng.choice(["Fully Paid", "Charged Off", "Current", "Default",
                         "Does not meet the credit policy. Status:Charged Off"], size=n,
                        p=[0.55, 0.25, 0.15, 0.03, 0.02])
    df = pd.DataFrame({
        "id": np.arange(1, n + 1).astype(str),
        "loan_amnt": rng.integers(1000, 35000, n).astype(float),
        "funded_amnt": rng.integers(1000, 35000, n).astype(float),
        "term": rng.choice([" 36 months", " 60 months"], size=n, p=[0.8, 0.2]),
        "int_rate": rng.uniform(5, 25, n), "installment": rng.uniform(50, 900, n),
        "grade": rng.choice(list("ABCDEFG"), size=n), "sub_grade": "A1",
        "emp_length": rng.choice(["10+ years", "< 1 year", "3 years", "1 year", None], size=n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], size=n),
        "annual_inc": rng.uniform(20000, 200000, n),
        "verification_status": rng.choice(["Verified", "Not Verified"], size=n),
        "issue_d": [f"{m}-{y}" for m, y in zip(months, years)],
        "loan_status": status,
        "purpose": rng.choice(["debt_consolidation", "credit_card", "car"], size=n),
        "addr_state": "CA", "dti": rng.uniform(0, 40, n), "delinq_2yrs": rng.integers(0, 3, n).astype(float),
        "earliest_cr_line": rng.choice(["Aug-2003", "Dec-1999", None], size=n),
        "fico_range_low": rng.integers(660, 800, n).astype(float),
        "inq_last_6mths": rng.integers(0, 5, n).astype(float), "open_acc": rng.integers(1, 30, n).astype(float),
        "pub_rec": rng.integers(0, 2, n).astype(float), "revol_bal": rng.uniform(0, 50000, n),
        "revol_util": rng.uniform(0, 100, n), "total_acc": rng.integers(2, 60, n).astype(float),
        "total_rec_prncp": rng.uniform(0, 30000, n), "recoveries": rng.uniform(0, 3000, n),
        "collection_recovery_fee": rng.uniform(0, 300, n), "out_prncp": 0.0, "last_pymnt_d": "Jan-2019",
        "application_type": "Individual", "policy_code": 1.0,
    })
    df["fico_range_high"] = df["fico_range_low"] + 4
    junk = pd.DataFrame({"id": ["Total amount funded in policy code 1: 123"], "issue_d": [None]})
    return pd.concat([df, junk], ignore_index=True)


@pytest.fixture
def raw_frame():
    return make_raw()
