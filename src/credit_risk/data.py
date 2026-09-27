import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk import config


def download(dest: Path = config.DATA_RAW) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / config.RAW_FILE
    if not target.exists():
        subprocess.run(["kaggle", "datasets", "download", config.KAGGLE_DATASET, "-f", config.RAW_FILE,
                        "-p", str(dest)], check=True)
    return target


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=config.RAW_COLUMNS, low_memory=False)


def parse_emp_length(s) -> float:
    if not isinstance(s, str):
        return np.nan
    if s.startswith("<"):
        return 0.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["issue_d"].notna() & pd.to_numeric(raw["id"], errors="coerce").notna()].copy()
    df = df[df["term"].str.strip() == f"{config.TERM_MONTHS} months"]
    df = df[df["loan_status"].isin(config.BAD_STATUSES | config.GOOD_STATUSES)]
    issue = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["vintage"] = issue.dt.year.astype(int)
    df["default"] = df["loan_status"].isin(config.BAD_STATUSES).astype(int)
    first_line = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (issue - first_line).dt.days / 365.25
    df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["emp_length_years"] = df["emp_length"].map(parse_emp_length)
    keep = (["id", "vintage", "default", config.BENCHMARK_COLUMN] + config.OUTCOME_COLUMNS
            + config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    return df[keep].reset_index(drop=True)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES].copy()


def build_interim(raw_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    clean(load_raw(raw_path)).to_parquet(out, index=False)
    return out
