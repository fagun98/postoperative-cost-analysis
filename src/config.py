"""Shared constants for the postoperative cost pilot."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

EXPECTED_ROWS = 2658
EXPECTED_COLS = 38

INDEX_CPT = "63280"
STUDY_START = pd.Timestamp("2026-01-01")
STUDY_END = pd.Timestamp("2026-07-31")

WINDOWS = [30, 60, 90]
DEFAULT_WINDOW = 90
DAY_BUCKETS = [(1, 30, "0-30"), (31, 60, "31-60"), (61, 90, "61-90")]

DATE_COLS = ["INDEX_SRVC_DT", "SRVC_STRT_DT", "SRVC_END_DT", "PROC_DT", "db"]
MONEY_COLS = ["CHRG_AMT", "COV_ALLW_CONTR_AMT", "NET_PD_AMT", "COINS_AMT", "COPAY_AMT", "DED_AMT"]
INT_COLS = ["DAYS_FROM_INDEX", "AGE_IN_YEARS", "ADJD_MED_CLM_SRVC_LN_NUM"]
DIAG_DROP = [f"DIAG{i}_CD" for i in range(2, 11)]

ALLOWED = "COV_ALLW_CONTR_AMT"

# Non-overlapping age groups (resolves the overlapping 55-65 / 65+ proposal).
AGE_GROUPS = [
    ("A", "A: <18", 0, 17),
    ("B", "B: 18-25", 18, 25),
    ("C", "C: 26-40", 26, 40),
    ("D", "D: 41-55", 41, 55),
    ("E", "E: 56-64", 56, 64),
    ("F", "F: 65+", 65, 200),
]
AGE_UNKNOWN = "Unknown"

TIN_WARNING = (
    "BILL_TIN is the source column `prov_id`, renamed as requested. "
    "It has not been verified as a tax ID."
)
