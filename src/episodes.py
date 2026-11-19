"""Build index episodes, attach later claim lines, and summarise each episode."""
import pandas as pd

from src import config

EPISODE_KEY = ["PERSON_KEY", "INDEX_SRVC_DT"]


def _mode(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else pd.NA


def build_index_episodes(claims, index_cpt=config.INDEX_CPT):
    """One row per (id+DOB, index date) with an index line for `index_cpt` in the study window."""
    idx = claims[
        (claims["JOURNEY_PHASE"] == "INDEX")
        & (claims["PROC_CD"] == index_cpt)
        & claims["INDEX_SRVC_DT"].between(config.STUDY_START, config.STUDY_END)
    ]
    data_cutoff = claims["PROC_DT"].max()

    rows = []
    for (person, index_dt), g in idx.groupby(EPISODE_KEY, sort=True):
        # The surgeon is taken as the rendering NPI on the highest-allowed index line.
        top = g.sort_values([config.ALLOWED, "rec"], ascending=[False, True]).iloc[0]
        rows.append({
            "PERSON_KEY": person,
            "id": g["id"].iloc[0],
            "INDEX_SRVC_DT": index_dt,
            "INDEX_CPT": index_cpt,
            "AGE_IN_YEARS": g["AGE_IN_YEARS"].iloc[0],
            "AGE_GROUP": g["AGE_GROUP"].iloc[0],
            "BILL_TIN": _mode(g["BILL_TIN"]),
            "BILL_PROV_NM": _mode(g["BILL_PROV_NM"]),
            "INDEX_REND_NPI": top["REND_NPI"],
            "INDEX_REND_NPIS": ";".join(sorted(g["REND_NPI"].dropna().unique())),
            "INDEX_DIAG1_CD": _mode(g["DIAG1_CD"]),
            "INDEX_DIAG1_DESC": _mode(g["DIAG1_DESC"]),
            "INDEX_LINES": len(g),
            "INDEX_ALLOWED": g[config.ALLOWED].sum(),
            "INDEX_ALL_DENIED": bool((g["PAID_DENIED"] == "DENIED").all()),
            "INDEX_MULTI_TIN": g["BILL_TIN"].nunique() > 1,
            "IDENTITY_CONFLICT": bool(g["IDENTITY_CONFLICT"].any()),
            "OBS_DAYS": (data_cutoff - index_dt).days,
        })
    ep = pd.DataFrame(rows)
    ep.insert(0, "EPISODE_ID", [f"E{i:03d}" for i in range(1, len(ep) + 1)])
    for w in config.WINDOWS:
        ep[f"COMPLETE_{w}"] = ep["OBS_DAYS"] >= w
    return ep


def attach_followup(claims, episodes):
    """Split each eligible episode's non-index lines into post-index, same-day and excluded."""
    lines = claims[claims["JOURNEY_PHASE"] != "INDEX"].merge(
        episodes[["EPISODE_ID"] + EPISODE_KEY], on=EPISODE_KEY, how="inner")
    in_2026 = lines["PROC_DT"].dt.year == 2026
    days = lines["DAYS_AFTER_SURGERY"]
    followup = lines[in_2026 & (days > 0)].sort_values(["EPISODE_ID", "SRVC_STRT_DT", "ADJD_MED_CLM_SRVC_LN_NUM", "rec"])
    sameday = lines[in_2026 & (days == 0)].sort_values(["EPISODE_ID", "rec"])
    excluded = lines[~in_2026 | (days < 0)]
    return followup.reset_index(drop=True), sameday.reset_index(drop=True), excluded


def episode_summary(episodes, followup, sameday, window=config.DEFAULT_WINDOW):
    """Per-episode index, postoperative (days 1..window) and total allowed amounts."""
    fu = followup[followup["DAYS_AFTER_SURGERY"] <= window]
    allowed = fu[config.ALLOWED]
    g = fu.assign(
        _paid=(fu["PAID_DENIED"] == "PAID") & (allowed > 0),
        _denied_zero=(fu["PAID_DENIED"] == "DENIED") | (allowed == 0),
    ).groupby("EPISODE_ID")
    agg = pd.DataFrame({
        "POSTOP_ALLOWED": g[config.ALLOWED].sum(),
        "POSTOP_LINES": g.size(),
        "POSTOP_PAID_LINES": g["_paid"].sum(),
        "POSTOP_DENIED_OR_ZERO_LINES": g["_denied_zero"].sum(),
        "POSTOP_DISTINCT_PROC": g["PROC_CD"].nunique(),
        "POSTOP_SERVICE_DAYS": g["SRVC_STRT_DT"].nunique(),
    })
    same = sameday.groupby("EPISODE_ID")[config.ALLOWED].agg(SAMEDAY_ALLOWED="sum", SAMEDAY_LINES="size")

    out = episodes.merge(agg, left_on="EPISODE_ID", right_index=True, how="left") \
                  .merge(same, left_on="EPISODE_ID", right_index=True, how="left")
    count_cols = ["POSTOP_LINES", "POSTOP_PAID_LINES", "POSTOP_DENIED_OR_ZERO_LINES",
                  "POSTOP_DISTINCT_PROC", "POSTOP_SERVICE_DAYS", "SAMEDAY_LINES"]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out[["POSTOP_ALLOWED", "SAMEDAY_ALLOWED"]] = out[["POSTOP_ALLOWED", "SAMEDAY_ALLOWED"]].fillna(0.0)
    out["TOTAL_ALLOWED"] = out["INDEX_ALLOWED"] + out["POSTOP_ALLOWED"]
    out["WINDOW_DAYS"] = window
    out["COMPLETE_FOLLOWUP"] = out["OBS_DAYS"] >= window
    return out


def build_all(claims, report):
    episodes = build_index_episodes(claims)
    followup, sameday, excluded = attach_followup(claims, episodes)
    summary = episode_summary(episodes, followup, sameday)

    extra = [report]
    if len(excluded):
        extra.append(pd.DataFrame({"issue_type": "excluded_from_episode", "rec": excluded["rec"].values,
                                   "id": excluded["id"].values, "detail": "before index or PROC_DT outside 2026"}))
    ep_flags = episodes.melt(id_vars=["EPISODE_ID", "id"],
                             value_vars=["INDEX_ALL_DENIED", "INDEX_MULTI_TIN", "IDENTITY_CONFLICT"])
    ep_flags = ep_flags[ep_flags["value"]]
    extra.append(pd.DataFrame({"issue_type": "episode_" + ep_flags["variable"].str.lower(), "rec": pd.NA,
                               "id": ep_flags["id"].values, "detail": "episode " + ep_flags["EPISODE_ID"].values}))
    incomplete = episodes[~episodes[f"COMPLETE_{config.DEFAULT_WINDOW}"]]
    extra.append(pd.DataFrame({"issue_type": "incomplete_followup", "rec": pd.NA, "id": incomplete["id"].values,
                               "detail": "episode " + incomplete["EPISODE_ID"] + f" observed < {config.DEFAULT_WINDOW} days"}))

    return {
        "claims_clean": claims,
        "index_episodes": episodes,
        "followup_claims": followup,
        "sameday_claims": sameday,
        "episode_summary": summary,
        "data_quality_report": pd.concat([e for e in extra if len(e)], ignore_index=True),
    }


def write_outputs(outputs, out_dir=config.PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        # `rec` stays only in the quality report, as the audit link back to source rows.
        if name != "data_quality_report":
            df = df.drop(columns=["rec"], errors="ignore")
        money = [c for c in df.columns if c in config.MONEY_COLS or c.endswith("_ALLOWED")]
        df = df.assign(**{c: df[c].round(2) for c in money})
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")


def read_outputs(out_dir=config.PROCESSED_DIR):
    """Read processed CSVs back with identifiers as text and dates parsed."""
    text_cols = {c: "string" for c in ["id", "PERSON_KEY", "BILL_TIN", "BILL_NPI", "BILL_MPIN", "REND_NPI",
                                       "INDEX_REND_NPI", "INDEX_REND_NPIS", "PROC_CD", "PLSRV_CD", "DIAG1_CD",
                                       "INDEX_DIAG1_CD", "INDEX_CPT", "EPISODE_ID"]}
    dates = ["INDEX_SRVC_DT", "SRVC_STRT_DT", "SRVC_END_DT", "PROC_DT", "db"]
    result = {}
    for name in ["index_episodes", "followup_claims", "sameday_claims", "episode_summary", "data_quality_report"]:
        df = pd.read_csv(out_dir / f"{name}.csv", dtype=text_cols)
        for col in dates:
            if col in df:
                df[col] = pd.to_datetime(df[col])
        result[name] = df
    return result
