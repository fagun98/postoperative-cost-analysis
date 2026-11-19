# Postoperative Cost Analysis POC — Implementation Plan

**Project:** Specialized Surgical Care — Postoperative Utilization, Cost Variation, and Provider Exploration  
**Pilot index procedure:** CPT `63280` (spinal surgery)  
**Scope:** Index surgeries performed **January 1–July 31, 2026**, with claims processed in 2026, as specified in the requirements  
**Version:** Planning baseline, November 19, 2026  
**Status:** Planning only. No preprocessing pipeline or Streamlit application has yet been implemented.

## 1. Project goal — plain language

Patients who have the **same main surgery** may receive different medical services afterward and have very different costs. We want to reconstruct each patient's care timeline, calculate the **insurance-allowed cost** of the surgery and subsequent services, and identify common or unusually costly patterns. Then we want a **simple Streamlit dashboard** to compare patients, procedures, diagnosis codes, and providers. **This pilot is not a quality-rating system:** high cost alone does not prove poor care.

### Pilot goals

1. Produce a repeatable, trustworthy dataset of CPT 63280 **index episodes** and subsequent claim lines.
2. Explain **what happened, when, how often, and at what allowed cost** after each surgery.
3. Group cases by observed **low / typical / elevated postoperative cost or utilization burden**, not “good/bad surgery.”
4. Compare billing organizations and rendering providers **descriptively**, with episode counts and limitations visible.
5. Present all of this in **native Streamlit components**, with no custom CSS.

### Explicitly **out of scope** for pilot v1

- Do **not** determine whether each later CPT code was caused by, or clinically related to, CPT 63280. Keep all later services in the initial exploratory measure.
- Do **not** attempt causal inference, risk-adjusted provider quality ratings, or predictive models.
- Do **not** develop a complex code ontology, clinical rules engine, diagnosis-to-procedure causal mapping, or ML clustering as a dependency for the first dashboard.
- Do **not** deploy the app to Databricks until John confirms the target environment. Make deployment portable.

## 2. Source data and current facts

**Source of truth:** `data.xlsx`. A direct, unfiltered conversion is supplied as `data_raw_lossless_ids.csv`. The user's previous `data.csv` export should **not** be used because it abbreviated distinct member IDs in scientific notation.

| Source verification | Result |
|---|---:|
| Excel worksheets | 1 |
| Data records | 2,658 |
| Columns | 38 |
| Distinct original `id` values | 21 |
| `JOURNEY_PHASE = INDEX` lines | 52 |
| `JOURNEY_PHASE = FOLLOW_UP` lines | 2,606 |
| Index lines within Jan–Jul 2026 | 33 |
| Distinct (`id`, `INDEX_SRVC_DT`) pairs in those index lines | 16, **provisional** |
| Follow-up lines with `DAYS_FROM_INDEX = 0` | 539 (all dates, not just eligible cohort) |
| `COV_ALLW_CONTR_AMT = 0` | 959 lines |
| `COV_ALLW_CONTR_AMT < 0` | 21 lines; investigate adjustments |

**Caution about IDs:** Three distinct `id` values are associated with conflicting `db` (DOB-like) values in the workbook. The 21 IDs are **not yet verified to represent 21 unique people**. Do not automatically merge those records. A (`id`, `INDEX_SRVC_DT`) episode key helps separate surgical events but does not by itself fix source identity collisions.

**Key source fields** (use the existing labels as evidence; rename only within the *processed* dataset):

- Identity and age: `id`, `AGE_IN_YEARS`, `db`
- Index and follow-up timing: `INDEX_SRVC_DT`, `SRVC_STRT_DT`, `SRVC_END_DT`, `DAYS_FROM_INDEX`, `JOURNEY_PHASE`, `PROC_DT`
- Services: `PROC_CD`, `PLSRV_CD`, `ADJD_MED_CLM_SRVC_LN_NUM`
- Costs and claims: `CHRG_AMT`, **`COV_ALLW_CONTR_AMT`**, `NET_PD_AMT`, `PAID_DENIED`
- Providers: `prov_id`, `BILL_NPI`, `BILL_MPIN`, `BILL_PROV_NM`, `REND_NPI`
- Diagnoses: `DIAG1_CD`, `DIAG1_DESC`, `DIAG2_CD`–`DIAG10_CD`

**Provider identity note:** Rename `prov_id` → `BILL_TIN` in the processed dataset **as explicitly requested by the project owner in this conversation**. The requirement document calls for Billing TIN, but **does not explicitly say that `prov_id` is the TIN**. Obtain confirmation or a mapping before reporting tax-ID-verified findings. No verified city/state or facility address is present in this workbook; `PLSRV_CD` is the **type of place of service**, not a geographic location.

## 3. Simple technical setup and file structure

### Task 0 — Create a local Python environment

**Subtasks**

- [ ] Create project directory and virtual environment.
- [ ] Install only necessary packages: `pandas`, `streamlit`, `pytest` (tests). Use Streamlit's built-in chart support; **no CSS, HTML injection, theme overrides, or custom visualization framework**.
- [ ] Keep input CSV immutable and generated files in separate folders.
- [ ] Save dependency versions after initial setup.

```bash
mkdir postoperative-care-poc
cd postoperative-care-poc
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas streamlit pytest
python -m pip freeze > requirements.lock.txt
```

**Recommended structure**

```text
postoperative-care-poc/
├── .venv/                            # local; do not commit
├── data/
│   ├── raw/
│   │   └── data_raw_lossless_ids.csv  # exact 38-column unfiltered source export
│   └── processed/
│       ├── claims_clean.csv
│       ├── index_episodes.csv
│       ├── followup_claims.csv
│       ├── episode_summary.csv
│       └── data_quality_report.csv
├── src/
│   ├── preprocess.py
│   ├── episodes.py
│   ├── metrics.py
│   └── app.py                       # Streamlit
├── tests/
│   ├── test_preprocessing.py
│   └── test_metrics.py
├── requirements.lock.txt
└── README.md
```

**Required dtype rule** — *read identifiers as text before transforming anything*:

```python
import pandas as pd

claims = pd.read_csv(
    "data/raw/data_raw_lossless_ids.csv",
    dtype={
        "id": "string",
        "prov_id": "string",
        "BILL_NPI": "string",
        "BILL_MPIN": "string",
        "REND_NPI": "string",
        "PROC_CD": "string",
        "DIAG1_CD": "string",
        "PLSRV_CD": "string",
    },
)
```

Parse dates explicitly with `pd.to_datetime(..., errors="coerce")`, then inspect parsing failures. Treat literal `null`, empty provider IDs, and whitespace consistently. Do not convert an identifier to a floating-point number. Be careful with opening/resaving CSV in spreadsheet applications, which can auto-format IDs.

## 4. Preprocessing tasks — `src/preprocess.py`

### Task 1 — Input integrity and type checks

- [ ] Read **all 2,658 records / 38 source columns**, before filtering. Keep source data intact.
- [ ] Assert required columns exist and `id` retains its **full original digit string**.
- [ ] Parse index/service/processed dates. Confirm `PROC_DT` means *claim process date*, not the date of the procedure performed.
- [ ] Convert money fields to numeric with explicit error reporting; retain negative and zero values for adjustment review.
- [ ] Standardize provider and code strings (`strip()` spaces, normalize literal `null` to missing, preserve identifiers as text).
- [ ] Save a small **quality report**: missing values, invalid dates/amounts, distinct counts, bad ID examples (redacted), and exceptions.

### Task 2 — Verify identity, episodes, and duplicate claims **before deleting anything**

- [ ] Inspect each `id` for conflicting `db`/DOB-like values and implausible ages; quarantine ambiguous records for review rather than merging them.
- [ ] Identify candidate index surgery lines with **`JOURNEY_PHASE == "INDEX"` and `PROC_CD == "63280"`**.
- [ ] Group multiple CPT 63280 claim lines into **candidate surgical episodes**, beginning with (`id`, `INDEX_SRVC_DT`) and reviewing provider/date conflicts.
- [ ] Check repeated lines and possible financial adjustments with `PAID_DENIED`, line numbers, dates, provider, CPT, and amounts. **Do not simply use `drop_duplicates()` across all columns.** The data does not expose a verified full claim ID sufficient to guarantee that identical-looking rows are duplicates.
- [ ] Log each excluded/combined row and the exact reason; avoid double-counting and preserve a reviewable exceptions report.
- [ ] Remove source column **`rec` from processed outputs**, but **only after** temporarily using it as an audit reference to trace source rows. Do not include it in final dashboard tables.

### Task 3 — Apply confirmed study-period filter

- [ ] Filter **index episodes** where `2026-01-01 <= INDEX_SRVC_DT <= 2026-07-31` (inclusive).
- [ ] Retain follow-up claim lines **for those eligible episodes**; do not independently filter every service by whether its service date is January–July.
- [ ] Follow the document's **2026 claim process date** requirement using `PROC_DT` if confirmed; distinguish the *claim process date* from `SRVC_STRT_DT`.
- [ ] Exclude claims that occur **before** the index date from postoperative cost, and treat **same-day claim lines** separately. The specification defines subsequent claims as service dates **later than** the index date. The workbook includes same-day rows labeled `FOLLOW_UP`.
- [ ] Derive `DAYS_AFTER_SURGERY = SRVC_STRT_DT - INDEX_SRVC_DT` from the actual dates and compare with `DAYS_FROM_INDEX`. Record disagreements.
- [ ] Keep available follow-up windows **30 / 60 / 90 days** configurable. For a fair comparison, flag episodes without enough observable follow-up/runout; **90 days is a proposed UI default, not a confirmed clinical rule**.

### Task 4 — Rename / derive analysis columns

- [ ] Rename `prov_id` → **`BILL_TIN`** as an **unverified source alias**, and document its source. Verify the mapping with John/client before using “tax ID” claims in reporting.
- [ ] Keep `BILL_NPI`, `REND_NPI`, and `BILL_PROV_NM` as separate provider attributes. `BILL_TIN` is an organization grouping, `REND_NPI` is a rendering-provider identifier, and **they should not be treated as interchangeable**.
- [ ] Build age groups from `AGE_IN_YEARS` with **no overlap**:

  | Group | Ages |
  |---|---|
  | A | Under 18 |
  | B | 18–25 |
  | C | 26–40 |
  | D | 41–55 |
  | E | 56–64 |
  | F | 65+ |

  *Note:* The originally suggested ranges “41–55, 55–65, 65+” overlap. This plan resolves them as above. Preserve `Unknown` for missing/invalid ages.
- [ ] **Drop `DIAG2_CD`–`DIAG10_CD` from the pilot's processed analysis table**, keeping `DIAG1_CD` and `DIAG1_DESC`. Keep the complete columns in the immutable raw source.
- [ ] Keep original `PROC_CD` as a **string**; values like `RX` or `MISC` exist, so do not assume every service code is a five-digit numeric CPT.
- [ ] Add `EPISODE_ID` for validated surgical episodes, not just a patient-level ID. If identity conflicts cannot be resolved, flag and exclude from definitive per-person summaries.

### Task 5 — Build simple, reusable output tables

- [ ] `claims_clean.csv`: cleaned line-level records (no `rec`, selected diagnosis fields, preserved monetary values and source fields).
- [ ] `index_episodes.csv`: one row per validated index surgery episode, with `EPISODE_ID`, index date, index CPT, patient age group, `BILL_TIN`, surgery `REND_NPI`, primary index diagnosis, and **index CPT 63280 allowed cost**.
- [ ] `followup_claims.csv`: all eligible **post-index-date** claim lines by episode, with follow-up days, `PROC_CD`, `DIAG1_CD`, provider IDs, and allowed amount. Keep unrelated possibilities included; no clinical-relatedness label.
- [ ] `episode_summary.csv`: per episode, index cost + follow-up allowed cost + total cost, follow-up count, distinct follow-up CPT count, and observation duration.
- [ ] `data_quality_report.csv`: ambiguous identities, candidate duplicates, zero/negative/denied/adjusted lines, inconsistent dates, and incomplete follow-up.

**Important metric rules**

- Use **`COV_ALLW_CONTR_AMT`** as the primary cost field; `CHRG_AMT` and `NET_PD_AMT` are different concepts.
- Define `TOTAL_EPISODE_ALLOWED = INDEX_CPT_63280_ALLOWED + POSTOPERATIVE_ALLOWED` for the selected window. Do not automatically add all day-zero service costs to the index CPT amount unless a separate, clearly labeled measure is approved.
- Handle negative adjustment/reversal lines consistently after reconciliation; do not delete them solely for being negative. Distinguish **denied/zero-allowed** services from paid utilization in charts.
- A claim line is not automatically a *unique visit*. Use exact claim/service definitions consistently and label outputs accordingly.

## 5. Simple relations to calculate — `src/episodes.py` and `src/metrics.py`

Focus on **descriptive relationships**, not clinical cause-and-effect. Derive aggregations after a validated episode table exists.

### Task 6 — Member and procedure relationships (required for v1)

1. **Episode → claim timeline**
   - [ ] Group by `EPISODE_ID`; sort by service date and claim line.
   - [ ] Measure service day, procedure code, allowed amount, and running postoperative total.
   - [ ] Output: one patient's chronological story, with optional 30/60/90-day selection.
2. **Procedure code → frequency and cost**
   - [ ] For each `PROC_CD`, report claim-line frequency, count of distinct episodes, median allowed amount/line, and total allowed amount.
   - [ ] Break out 0–30, 31–60, and 61–90 days, if usable; highlight high-volume/high-cost services.
3. **Episode → postoperative cost burden**
   - [ ] Compute index allowed, follow-up allowed, total allowed, and follow-up lines per episode.
   - [ ] Report median, average, min/max, percentiles, and distribution, including zero/negative adjustments correctly.
4. **Age group → cost and utilization**
   - [ ] Compare episode count, median postoperative cost, and follow-up frequency by A–F.
   - [ ] Age is descriptive context, **not adequate risk adjustment**.
5. **Primary claim diagnosis → cost / services**
   - [ ] Compare `DIAG1_CD` and follow-up `PROC_CD` combinations, including which billing organizations submit them and what their allowed amounts look like.
   - [ ] Also offer a separate **index surgery `DIAG1_CD`** filter where possible. The follow-up claim's diagnosis can differ from the index diagnosis, so do not assume both describe the same illness.
6. **Billing organization → rendering provider**
   - [ ] Group by provisional `BILL_TIN` → `REND_NPI` and report volumes and allowed amounts for **services those entities billed/rendered**.
   - [ ] For **episode outcome comparisons**, assign the episode to its **index surgery billing entity**; do not attribute a patient's entire downstream spending to whichever specialist billed an unrelated follow-up line.

### Task 7 — Simple exploratory segmentation and adjustable thresholds

- [ ] Start with **lower / typical / elevated postoperative cost burden**, derived from a selected window's observed distribution (e.g., percentiles) — **not “good / bad provider.”**
- [ ] Add an optional **dollar threshold** (or percentile threshold) to flag episodes above a selected postoperative allowed cost. Display what fraction of cases exceed it.
- [ ] Compare **episode count, median postoperative allowed, index allowed, follow-up lines, and high-cost share** among groups.
- [ ] At the billing organization level, show how many of its **index surgical episodes** fall in the elevated-burden group. Always show sample size; do not rank one-episode organizations as reliably better or worse.
- [ ] Enable a drill-down from a billing organization to its rendering NPIs, while distinguishing **who performed the index surgery** from who billed later care.
- [ ] Treat thresholds as **investigation flags**, not clinically validated cutoffs. Recompute group counts as thresholds change.

### Task 8 — Optional extra findings, only if the data supports them

- [ ] Index allowed **versus** later allowed cost: does a more expensive initial surgery sometimes have less observed downstream spending? Association only.
- [ ] Same **`PROC_CD` + `DIAG1_CD`** across providers: display median allowed **and sample size**. Show comparisons only where groups are comparable; different case complexity and payment contracts may explain prices.
- [ ] Index surgery **`BILL_TIN` → `REND_NPI`**: which rendering providers account for the most episodes or highest observed follow-up burden within a billing organization? **No blame without clinical review.**
- [ ] Service setting (`PLSRV_CD`), in-network/out-of-network (`INN_OON_CD`), and claim status (`PAID_DENIED`) as optional explanatory filters.
- [ ] Sensitivity check: do observed rankings/cohorts change dramatically across 30, 60, and 90 days, or when adjustment candidates are handled differently?
- [ ] Show top five **highest / lowest descriptive median costs**, *only if the group sizes are adequate*, and label them **cost comparisons**, not provider quality rankings.

## 6. Streamlit dashboard — minimal, default components only

### Task 9 — Build `src/app.py` with **four tabs**

**Global controls:** `st.sidebar.selectbox` for index CPT (initially only `63280`), `st.sidebar.select_slider` for 30/60/90-day window, filters for age group / index diagnosis / billing organization, and one editable **postoperative allowed-cost threshold**. Use `st.metric`, `st.dataframe`, `st.tabs`, `st.selectbox`, `st.slider`/`st.number_input`, `st.bar_chart`, `st.line_chart`, and `st.scatter_chart`. No custom CSS, custom components, complex charts, or overloaded screens.

| Tab | What the user sees | Streamlit visualization |
|---|---|---|
| **1. Overview** | Number of valid index episodes; median surgery, follow-up and total allowed costs; costs by age group and billing organization | `st.metric`; binned cost distribution `st.bar_chart`; compact summary `st.dataframe` |
| **2. Care Patterns** | Most common follow-up procedure codes; number of people receiving each; costs and timing by procedure | Horizontal/vertical `st.bar_chart`; 0–30/31–60/61–90 summary; sortable `st.dataframe` |
| **3. Cohorts & Thresholds** | Lower / typical / elevated-burden cohorts; adjustable allowed-cost cutoff; how counts change | `st.number_input` or `st.slider`; `st.metric`; `st.bar_chart` |
| **4. Member & Provider Explorer** | Choose a de-identified episode to see day-by-day claims and running cost; choose billing org to see rendering providers and their volume/cost | `st.selectbox`; chronological `st.dataframe`; `st.line_chart`; grouped provider `st.dataframe` |

**Keep charts understandable**

- Always label **allowed amount**, time window, and whether a number counts **claim lines, episodes, or distinct people**.
- Limit top-code bar charts to **top 10**; put the complete list in a filterable table.
- Keep unrelated/unknown-clinical-relevance claims in the initial total and label that scope clearly.
- Show a visible **“small sample / exploratory only”** notice on provider views. Do **not** call organizations or doctors “good” or “bad.”
- Add a collapsed `st.expander("Data quality and assumptions")` for excluded rows, identity conflicts, adjustments, and follow-up coverage; no fifth full tab necessary.
- Avoid styling: use Streamlit's **default theme, native layout, no CSS files, and no `st.markdown(..., unsafe_allow_html=True)`**.
- Build the UI with CSV inputs first; if John confirms client-side Databricks deployment, adapt file access and deployment as a separate task.

### Task 10 — Verification and delivery

- [ ] **Conversion QA:** every original `id` exactly matches the new raw CSV, 2,658 rows, 38 columns, same column ordering and values; verify with CSV read-back.
- [ ] **Preprocessing QA:** all identifiers are text; only index episodes Jan–Jul 2026 eligible; all included follow-up dates strictly after surgery; `rec` and `DIAG2_CD`–`DIAG10_CD` absent in processed claims only.
- [ ] **Cost QA:** hand-check 3–5 episodes against source lines; ensure that `index + postoperative = total` and adjustment/denial handling is consistent.
- [ ] **Cohort QA:** a selected cutoff changes flagged case counts without silently changing the underlying claims; no duplicated episodes.
- [ ] **Provider QA:** show `BILL_TIN` mapping warning until verified; preserve `REND_NPI` distinction; include counts alongside cost values.
- [ ] **UI QA:** run `streamlit run src/app.py`; all four tabs work with default components, filters update tables/charts, no CSS/custom HTML, friendly warnings for empty filters.
- [ ] **Pilot review:** discuss findings, cohort cutoffs, TIN/CAE definitions, sample sizes, and optional Databricks deployment with John.

**Done for v1 when:** a nontechnical reviewer can select a patient episode, see its follow-up timeline and allowed costs, compare the most common procedures, adjust a cost-burden cutoff, and inspect billing/rendering-provider patterns with sample-size warnings.

## 7. Risks and decisions awaiting confirmation

1. **Billing TIN:** `prov_id` → `BILL_TIN` is a **user-specified mapping**, not independently verified. Ask the client to confirm before calling it a tax ID in final reports.
2. **Patient identity:** 3 IDs have conflicting DOB-like values. Do not silently collapse them into unique individuals; ask for a verified episode/member key or clinical/source guidance.
3. **Duplicate and financial adjustments:** 21 negative allowed lines and 959 zero-allowed lines exist. Reconcile denied/adjustment lines rather than simply dropping them.
4. **Same-day follow-ups:** 539 follow-up-labeled lines have `DAYS_FROM_INDEX = 0` (full raw file); specification says *later than index date*. Show same-day care separately until clarified.
5. **Follow-up completeness:** July surgeries need sufficient observation time and claims runout; a later process date does not guarantee complete utilization.
6. **Very small provider samples:** about **16 candidate eligible (`id`, index date) episode pairs across 15 index billing IDs** before full validation, making top-five quality rankings misleading.
7. **Geography/CAE:** no explicit location or `CAE` field is evident among the spreadsheet headers. Obtain the missing dimension/mapping if required.
8. **Clinical relevance:** deliberately defer linking later services medically to CPT 63280. That is a later clinical-review enhancement, not a pilot preprocessing rule.

---

## 8. Evidence and references — why each rule exists

This section deliberately distinguishes **source requirements**, **user instructions**, **data observations**, and **implementation proposals**. No external clinical rules or causal claims are being assumed.

| Rule / decision | Evidence / authority | Interpretation |
|---|---|---|
| Focus on index CPT `63280` | **Requirements DOCX**, section 1 (p. 1), section 2.1 (p. 2) | One main surgery type for pilot; later claim codes vary. |
| **Jan 1–Jul 31, 2026** for index surgeries | **John's follow-up email supplied in chat**: limit index surgeries to this period for comparable downstream observation | Confirmed index-date restriction; follow-up services **are not** restricted to the same seven months. |
| Use **2026 claim process dates** | **Requirements DOCX**, section 2.1 (p. 2) and Data considerations (p. 4) | Use `PROC_DT` as process date after validating source meaning. Independent of index-surgery-date rule. |
| Use `INDEX_SRVC_DT` for episode filter | **Requirements DOCX**, section 2.2 (p. 2) — index surgery date of service; **source workbook** has `INDEX_SRVC_DT` | This field identifies the primary surgery date in the supplied dataset. |
| Retrieve all later CPT-coded claims **without clinical prefiltering** | **Requirements DOCX**, section 2.3 (pp. 2–3), section 3 Analysis B (p. 5) | Include later services first, even when unrelatedness is possible. User explicitly postponed clinical linkage for pilot. |
| Subsequent service timing | **Requirements DOCX**, section 2.3 (p. 2) — service date *later than* index date | Distinguish service dates greater than the index from same-day/perioperative service lines. |
| **`COV_ALLW_CONTR_AMT`** as cost metric | **Requirements DOCX**, sections 2.2 and 3 (pp. 2, 4–5) require *allowed amount*; **source workbook** has `COV_ALLW_CONTR_AMT` | Correct matching raw column for index, postoperative, total allowed cost. Not proof of provider quality. |
| `prov_id` → **`BILL_TIN`** | **Explicit user instruction** in this chat; **requirements DOCX**, sections 2.1 and 3 (pp. 2, 4) require Billing TIN; **source workbook** provides `prov_id` but no named TIN field | Perform requested rename **provisionally**; source does **not** prove `prov_id` is the billing tax ID. Validate mapping. |
| Drop `rec`, retain only `DIAG1_CD` in processed pilot data | **Explicit user instructions** in this chat | Remove after audit; don't change full raw export. Also retain `DIAG1_DESC` for display. |
| Age groups A–F | **Explicit user instruction** in this chat | Adjust overlapping original boundary to A <18, B 18–25, C 26–40, D 41–55, E 56–64, F 65+. |
| Compare index and downstream costs, CPT patterns, providers | **Requirements DOCX**, sections 3 and 4 (pp. 4–8) | Descriptive distributions, utilization, exploratory cohorts; **not** causal quality scoring. |
| UI capabilities and nonproduction POC | **Requirements DOCX**, section 5 (pp. 8–11) and John's original supervisor email quoted in chat | Procedure overview, care patterns, cohorts, member journey, provider comparison. UI should support future CPT codes. |
| **Streamlit, no CSS, default components** | **Explicit user instruction** in this chat | Implementation choice; not a requirement from John. |
| Client-facing Databricks deployment | **Question in draft email to John**, not yet answered | Keep as an open choice; do not assume deployment or add Databricks dependencies for v1. |
| ID precision in raw CSV | **`data.xlsx` stored OOXML cell values**, verified against generated `data_raw_lossless_ids.csv` | All 2,658 ID cells match exactly; original 21 distinct values preserved as plain digit strings. |
| Identity collisions, same-day follow-ups, financial adjustments, tiny sample | **Direct inspection of `data.xlsx` / lossless CSV** | These are QA findings requiring explicit handling; they are not judgments of care quality. |

**Sources supplied for this plan:** `Postoperative Care Quality and Cost Analysis POC.docx` (12 pages); `data.xlsx` (1 sheet, 38 columns); John's initial supervisor email; John's follow-up message about index-date limits; and subsequent user decisions in this conversation.

**Conversion note:** `data_raw_lossless_ids.csv` is **raw and unfiltered**. It retains every source column and the literal stored numeric value for all ordinary numeric cells; Excel date serials were exported as equivalent ISO `YYYY-MM-DD` dates. CSV cannot preserve Excel-specific formatting or cell types, so programmatic imports must explicitly treat `id` and all provider/code identifiers as strings. No Excel formulas were present in the source sheet.
