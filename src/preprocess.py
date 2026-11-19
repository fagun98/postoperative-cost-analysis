"""Load, type-check, quality-check and clean the raw claims export.

Run the whole pipeline with:  python -m src.preprocess
"""
import pandas as pd

from src import config


def load_raw(path=config.RAW_CSV):
    """Read every column as text first so identifiers are never coerced to floats."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert raw.shape == (config.EXPECTED_ROWS, config.EXPECTED_COLS), raw.shape

    df = raw.apply(lambda s: s.str.strip())
    df = df.mask(df.isin(["", "null", "NULL"]))

    for col in config.DATE_COLS:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    for col in config.MONEY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["rec"] = pd.to_numeric(df["rec"]).astype(int)
    return df, raw


def age_group(age):
    if pd.isna(age):
        return config.AGE_UNKNOWN
    for _, label, lo, hi in config.AGE_GROUPS:
        if lo <= age <= hi:
            return label
    return config.AGE_UNKNOWN


def _issue(rows, issue_type, detail):
    return pd.DataFrame({
        "issue_type": issue_type,
        "rec": rows["rec"].values,
        "id": rows["id"].values,
        "detail": detail if isinstance(detail, str) else list(detail),
    })


def quality_checks(df, raw):
    """Return one row per flagged source line. Nothing is dropped here."""
    issues = []

    for col in config.DATE_COLS + config.MONEY_COLS + config.INT_COLS:
        bad = df[df[col].isna() & ~raw[col].str.strip().isin(["", "null", "NULL"])]
        if len(bad):
            issues.append(_issue(bad, "unparseable_value", f"{col} could not be parsed"))
    for col in ["id", "INDEX_SRVC_DT", "SRVC_STRT_DT", "PROC_CD", config.ALLOWED, "BILL_NPI", "prov_id"]:
        miss = df[df[col].isna()]
        if len(miss):
            issues.append(_issue(miss, "missing_value", f"{col} is missing"))

    allowed = df[config.ALLOWED]
    issues.append(_issue(df[allowed < 0], "negative_allowed", allowed[allowed < 0].map("allowed={:.2f}".format)))
    issues.append(_issue(df[allowed == 0], "zero_allowed", "allowed=0 (" + df.loc[allowed == 0, "PAID_DENIED"].fillna("?") + ")"))
    issues.append(_issue(df[df["PAID_DENIED"] == "FINADJ"], "financial_adjustment", "PAID_DENIED=FINADJ"))

    derived = (df["SRVC_STRT_DT"] - df["INDEX_SRVC_DT"]).dt.days
    mismatch = df[derived != df["DAYS_FROM_INDEX"]]
    issues.append(_issue(mismatch, "days_from_index_mismatch",
                         "derived=" + derived[mismatch.index].astype(str) + " source=" + mismatch["DAYS_FROM_INDEX"].astype(str)))
    sameday = df[(derived == 0) & (df["JOURNEY_PHASE"] == "FOLLOW_UP")]
    issues.append(_issue(sameday, "sameday_followup", "FOLLOW_UP line on the index date; reported separately"))
    before = df[derived < 0]
    issues.append(_issue(before, "service_before_index", "service date before index date; excluded"))

    dob_counts = df.groupby("id")["db"].nunique()
    conflict_ids = dob_counts[dob_counts > 1].index
    conflicts = df[df["id"].isin(conflict_ids)]
    issues.append(_issue(conflicts, "identity_conflict",
                         "id has " + conflicts["id"].map(dob_counts).astype(str) + " DOB values; split by id+DOB, not merged"))

    dup_key = ["id", "INDEX_SRVC_DT", "SRVC_STRT_DT", "PROC_CD", "ADJD_MED_CLM_SRVC_LN_NUM",
               "BILL_NPI", "REND_NPI", config.ALLOWED]
    dups = df[df.duplicated(dup_key, keep=False)]
    issues.append(_issue(dups, "candidate_duplicate", "identical on key service fields; kept, review needed"))

    rev_key = ["id", "INDEX_SRVC_DT", "SRVC_STRT_DT", "PROC_CD", "BILL_NPI"]
    neg = df[allowed < 0]
    pos = df[allowed > 0].assign(_abs=lambda d: d[config.ALLOWED])
    pairs = neg.assign(_abs=-neg[config.ALLOWED]).merge(pos[rev_key + ["_abs", "rec"]], on=rev_key + ["_abs"], suffixes=("", "_orig"))
    if len(pairs):
        issues.append(_issue(pairs, "reversal_pair", "offsets rec " + pairs["rec_orig"].astype(str)))

    report = pd.concat([i for i in issues if len(i)], ignore_index=True)
    return report.sort_values(["issue_type", "rec"]).reset_index(drop=True)


def clean(df):
    """Rename/derive analysis columns. `rec` is kept here and dropped at write time."""
    out = df.rename(columns={"prov_id": "BILL_TIN"}).drop(columns=config.DIAG_DROP)
    out["AGE_GROUP"] = out["AGE_IN_YEARS"].map(age_group)
    out["DAYS_AFTER_SURGERY"] = (out["SRVC_STRT_DT"] - out["INDEX_SRVC_DT"]).dt.days
    out["PERSON_KEY"] = out["id"] + "|" + out["db"].dt.strftime("%Y-%m-%d").fillna("NA")
    dob_counts = out.groupby("id")["db"].nunique()
    out["IDENTITY_CONFLICT"] = out["id"].map(dob_counts) > 1
    return out


def main():
    from src import episodes

    df, raw = load_raw()
    report = quality_checks(df, raw)
    claims = clean(df)
    outputs = episodes.build_all(claims, report)
    episodes.write_outputs(outputs)

    ep = outputs["index_episodes"]
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols; {df['id'].nunique()} distinct ids, "
          f"{claims['PERSON_KEY'].nunique()} id+DOB keys")
    print(f"Eligible index episodes (CPT {config.INDEX_CPT}, {config.STUDY_START.date()}..{config.STUDY_END.date()}): {len(ep)}")
    print(f"Follow-up lines (day>0): {len(outputs['followup_claims']):,}; same-day lines: {len(outputs['sameday_claims']):,}")
    print("Quality report counts:")
    print(outputs["data_quality_report"]["issue_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
