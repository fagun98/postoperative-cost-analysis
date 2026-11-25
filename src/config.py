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

# Place-of-service codes used to bound the index hospitalization and major events.
INPATIENT_POS = "21"
OUTPATIENT_SURGERY_POS = frozenset({"22", "24"})

# Heuristic spine/neuro family for major-event relatedness (not clinical adjudication).
# Related if event TOP_DIAG1 matches index DIAG1, starts with a prefix below, is in
# RELATED_DIAG_CODES, or TOP_PROC_CD is the index CPT / in RELATED_PROC_CODES.
RELATED_DIAG_PREFIXES = frozenset({
    "D32",   # benign neoplasm of meninges
    "D33",   # benign neoplasm of brain/CNS
    "D36",   # benign neoplasm of peripheral nerves / ANS
    "C47",   # malignant neoplasm peripheral nerves
    "C71",   # malignant neoplasm of brain
    "C72",   # malignant neoplasm of spinal cord / CNS
    "G95",   # other diseases of spinal cord
    "M54",   # dorsalgia / radiculopathy
})
RELATED_DIAG_CODES = frozenset({
    "D492",  # neoplasm unspecified behavior, bone/soft tissue (seen as index DIAG1)
    "D497",  # neoplasm unspecified behavior, endocrine/other NS
})
RELATED_PROC_CODES = frozenset({
    INDEX_CPT,  # repeat of index spine surgery CPT
    "63001", "63003", "63005", "63012", "63015", "63016", "63017",
    "63020", "63030", "63035", "63040", "63042", "63045", "63046", "63047",
    "63048", "63050", "63051", "63055", "63056", "63057", "63064", "63066",
    "63075", "63076", "63077", "63078", "63081", "63082", "63085", "63086",
    "63087", "63088", "63090", "63091", "63101", "63102", "63103",
})

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
