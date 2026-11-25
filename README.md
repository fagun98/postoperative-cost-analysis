# Postoperative Cost Analysis — Pilot (CPT 63280)

Exploratory pipeline and Streamlit dashboard for postoperative utilization and insurance-allowed cost after CPT 63280 index surgeries performed 2026-01-01 to 2026-07-31. See `pilot_plan.md` for the full specification. This is descriptive only; it is not a quality rating.

## Setup

```bash
/Users/fagun/anaconda3/bin/python3.11 -m venv .venv    # any Python >= 3.10
.venv/bin/python -m pip install -r requirements.lock.txt
```

## Run

```bash
.venv/bin/python -m src.preprocess      # data/raw/data.csv -> data/processed/*.csv
.venv/bin/python -m pytest -q           # unit tests
.venv/bin/python scripts/validate_sample_stays.py  # E001-E010 stay summary for review
.venv/bin/streamlit run src/app.py      # dashboard
```

## Deploy on Databricks Apps

1. Ensure `data/processed/*.csv` exists (run preprocess first if needed).
2. Create a Databricks App in the workspace and point it at this project root.
3. `app.yaml` starts Streamlit with `src/app.py` on `$DATABRICKS_APP_PORT`.
4. `requirements.txt` is what the Apps runtime installs (`streamlit`, `pandas`).

This pilot still reads local processed CSVs bundled with the app. It does not yet query Unity Catalog or a SQL warehouse.

## Layout

| Path | Purpose |
|---|---|
| `data/raw/data.csv` | Unmodified copy of the supplied `data.csv` (2,658 rows × 38 cols) |
| `src/config.py` | Study window, CPT, follow-up windows, age groups, column lists |
| `src/preprocess.py` | Text-safe load, type parsing, quality checks, renames/derived columns |
| `src/episodes.py` | Stay-based index episodes, post-discharge follow-up, major events, CSV I/O |
| `src/metrics.py` | Descriptive aggregations, cohorts, thresholds, window sensitivity |
| `src/app.py` | Two-tab dashboard (Overview, Drill Down), native Streamlit components only |
| `scripts/validate_sample_stays.py` | Builds the E001–E010 stay / event summary table |
| `data/processed/` | `claims_clean`, `index_episodes`, `index_stay_claims`, `followup_claims`, `major_events`, `episode_summary`, `data_quality_report` |
| `app.yaml` | Databricks Apps start command and Streamlit proxy env |
| `requirements.txt` | Runtime deps for Databricks Apps deploy |

## Definitions and assumptions

- **Episode key:** INDEX-phase CPT 63280 lines grouped by member `id` + DOB (`db`) + `INDEX_SRVC_DT`. There are 16 eligible episodes.
- **Index episode (hospital stay):** When POS `21` is present on the surgery date, the index is the contiguous inpatient stay (all claim lines from stay start through discharge). If there is no POS 21 on the surgery date, the index is all surgery-day claims. `INDEX_ALLOWED` is the stay total, not CPT-only.
- **Postoperative:** claims with a service date after discharge, within the 30/60/90-day window, and a 2026 process date. Windows are counted from `STAY_END`.
- **Major events:** after discharge, contiguous POS 21 runs (subsequent inpatient) and POS 22/24 days (outpatient surgery/facility) are flagged as investigation signals, not quality scores. Each event is labeled **Related** or **Unrelated** to the index spine surgery using a deterministic spine/neuro DIAG1 prefix / CPT family list (plus exact index DIAG1 or index CPT match). This is a heuristic for investigation, not clinical adjudication.
- **Cost:** `COV_ALLW_CONTR_AMT` (allowed amount). `TOTAL_ALLOWED = INDEX_ALLOWED + POSTOP_ALLOWED`. Cost statistics omit denied and zero-allowed amounts.
- **Index provider:** `BILL_TIN` is the most common TIN on the CPT index lines. `INDEX_REND_NPI` is the rendering NPI on the highest-allowed CPT index line.
- **Observation:** `OBS_DAYS` is the latest `PROC_DT` minus discharge (`STAY_END`).
- **Age groups:** A <18, B 18–25, C 26–40, D 41–55, E 56–64, F 65+.
- `rec` is removed from processed tables but kept in `data_quality_report.csv` for tracing. `DIAG2_CD`–`DIAG10_CD` are dropped from processed tables.

## Data findings to be aware of

- **Member IDs were truncated in the source workbook itself.** The `id` values in `docs/data.xlsx` (local only) match `data.csv`. Three ids each carry two different DOBs and are split by DOB (`IDENTITY_CONFLICT`).
- Cost statistics exclude denied claims and zero-allowed amounts. Episodes with no eligible index-stay dollars can also be filtered out in the dashboard.
- 327 lines are candidate duplicates and 17 negative lines look like reversals of an earlier line. Both are flagged, not removed.

## Open questions

1. Is `prov_id` the billing TIN? It is renamed `BILL_TIN` provisionally.
2. Can the source supply untruncated member IDs to resolve the 3 identity collisions?
3. Is `PROC_DT` the claim process date?
4. Should long unbroken POS 21 runs (for example an 85-day stay) be capped or split?
5. Is the deployment target Databricks? The app currently reads local CSVs.
