from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

KAGGLE_DATASET = "wordsforthewise/lending-club"
RAW_FILE = "accepted_2007_to_2018Q4.csv.gz"

BAD_STATUSES = frozenset({"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"})
GOOD_STATUSES = frozenset({"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"})

TERM_MONTHS = 36
TRAIN_YEARS = tuple(range(2007, 2014))   # 2007-2013
OOT_YEARS = (2014, 2015)                 # fully matured 36-month vintages
HOLDOUT_FRAC = 0.2
SEED = 42

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "emp_length_years", "fico_mid", "inq_last_6mths", "delinq_2yrs",
    "pub_rec", "open_acc", "total_acc", "revol_bal", "revol_util", "credit_history_years",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
BENCHMARK_COLUMN = "grade"
OUTCOME_COLUMNS = ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]
LEAKAGE_COLUMNS = frozenset({"loan_status", "total_rec_prncp", "recoveries", "collection_recovery_fee",
                             "out_prncp", "last_pymnt_d"})

RAW_COLUMNS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status", "purpose", "dti",
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "total_rec_prncp", "recoveries",
    "collection_recovery_fee", "out_prncp", "last_pymnt_d",
]

PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50
