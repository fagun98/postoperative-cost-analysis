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
.venv/bin/python -m pytest -q           # 18 tests
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
| `src/episodes.py` | Index episodes, follow-up/same-day split, per-episode summary, CSV I/O |
| `src/metrics.py` | Descriptive aggregations, cohorts, thresholds, window sensitivity |
| `src/app.py` | Four-tab dashboard, native Streamlit components only |
| `data/processed/` | `claims_clean`, `index_episodes`, `followup_claims`, `sameday_claims`, `episode_summary` (90-day), `data_quality_report` |
| `app.yaml` | Databricks Apps start command and Streamlit proxy env |
| `requirements.txt` | Runtime deps for Databricks Apps deploy |

## Definitions and assumptions

- **Episode:** INDEX-phase CPT 63280 lines grouped by member `id` + DOB (`db`) + `INDEX_SRVC_DT`. There are 16 eligible episodes.
- **Cost:** `COV_ALLW_CONTR_AMT` (allowed amount). `TOTAL_ALLOWED = INDEX_ALLOWED + POSTOP_ALLOWED`.
- **Postoperative:** non-index lines with a service date 1 to *window* days after the index date (window 30/60/90) and a 2026 process date. Denied, zero and FINADJ (negative) lines are kept and netted, never dropped.
- **Same-day lines** (FOLLOW_UP on the index date) are reported separately (`SAMEDAY_ALLOWED`) and are not part of the total.
- **Index provider:** `BILL_TIN` is the most common TIN on the index lines. `INDEX_REND_NPI` is the rendering NPI on the highest-allowed index line. All index NPIs are listed in `INDEX_REND_NPIS`.
- **Observation:** `OBS_DAYS` is the latest `PROC_DT` in the file minus the index date. All 16 episodes have at least 104 days, so every window is observable, though claims runout may still be incomplete.
- **Age groups:** A <18, B 18–25, C 26–40, D 41–55, E 56–64, F 65+.
- `rec` is removed from processed tables but kept in `data_quality_report.csv` for tracing. `DIAG2_CD`–`DIAG10_CD` are dropped from processed tables.

## Data findings to be aware of

- **Member IDs were truncated in the source workbook itself.** The `id` values in `docs/data.xlsx` are stored as `116219000000` and similar, identical to `data.csv` (covered by a test). Three ids (`116218000000`, `116219000000`, `116225000000`) each carry two different DOBs. They are split by DOB and flagged `IDENTITY_CONFLICT`; 5 episodes are affected.
- **8 of 16 episodes have every index 63280 line denied**, so their `INDEX_ALLOWED` is $0. Cost statistics (mean, median, percentiles, cohorts, thresholds) exclude denied claims and zero-allowed amounts, so those episodes do not pull the median down. The dashboard also has a filter to drop them from the selected cohort entirely.
- 327 lines are candidate duplicates and 17 negative lines look like reversals of an earlier line. Both are flagged, not removed.

## Open questions

1. Is `prov_id` the billing TIN? It is renamed `BILL_TIN` provisionally.
2. Can the source supply untruncated member IDs to resolve the 3 identity collisions?
3. Is `PROC_DT` the claim process date?
4. Should same-day lines count toward the index or the postoperative cost?
5. How should episodes with a denied index surgery be treated?
6. Is the deployment target Databricks? The app currently reads local CSVs.
