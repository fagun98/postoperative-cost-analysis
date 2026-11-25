"""Descriptive aggregations over episodes and follow-up lines. No causal or quality scoring."""
import pandas as pd

from src import config

COHORTS = ["Lower", "Typical", "Elevated"]
DAY_COL = "DAYS_AFTER_DISCHARGE"


def _day_col(df):
    if DAY_COL in df.columns:
        return DAY_COL
    return "DAYS_AFTER_SURGERY"


def in_window(followup, window):
    col = _day_col(followup)
    return followup[followup[col] <= window]


def cost_eligible_mask(df):
    """Lines that count toward cost statistics: not denied, and allowed amount > 0."""
    return (df["PAID_DENIED"] != "DENIED") & (df[config.ALLOWED] > 0)


def cost_eligible_lines(df):
    return df[cost_eligible_mask(df)]


def cost_stat_series(summary, col="POSTOP_ALLOWED"):
    """Episode amounts used for mean/median/distribution. Ineligible episodes are NaN."""
    s = summary[col].astype(float)
    if col == "INDEX_ALLOWED":
        if "INDEX_PAID_LINES" in summary.columns:
            return s.where(summary["INDEX_PAID_LINES"] > 0)
        return s.where(~summary["INDEX_ALL_DENIED"] & (s > 0))
    if col == "POSTOP_ALLOWED":
        return s.where(summary["POSTOP_PAID_LINES"] > 0)
    if col == "TOTAL_ALLOWED":
        if "INDEX_PAID_LINES" in summary.columns:
            index_ok = summary["INDEX_PAID_LINES"] > 0
        else:
            index_ok = ~summary["INDEX_ALL_DENIED"] & (summary["INDEX_ALLOWED"] > 0)
        postop_ok = summary["POSTOP_PAID_LINES"] > 0
        return s.where(index_ok | postop_ok)
    return s


def day_bucket(days):
    for lo, hi, label in config.DAY_BUCKETS:
        if lo <= days <= hi:
            return label
    return ">90"


def timeline(followup, episode_id, window):
    """One episode's post-discharge lines in date order with a running allowed total."""
    col = _day_col(followup)
    t = in_window(followup[followup["EPISODE_ID"] == episode_id], window)
    t = t.sort_values(["SRVC_STRT_DT", "ADJD_MED_CLM_SRVC_LN_NUM"]).copy()
    t["RUNNING_ALLOWED"] = t[config.ALLOWED].cumsum()
    cols = ["SRVC_STRT_DT", col, "PROC_CD", "DIAG1_CD", "DIAG1_DESC", "PLSRV_CD",
            "BILL_TIN", "BILL_PROV_NM", "REND_NPI", "PAID_DENIED", config.ALLOWED, "RUNNING_ALLOWED"]
    return t[[c for c in cols if c in t.columns]].reset_index(drop=True)


def proc_code_summary(followup, window):
    """Per PROC_CD: claim lines, distinct episodes, allowed stats and allowed by day bucket."""
    col = _day_col(followup)
    fu = in_window(followup, window).assign(BUCKET=lambda d: d[col].map(day_bucket))
    eligible = cost_eligible_lines(fu)
    g = fu.groupby("PROC_CD")
    eg = eligible.groupby("PROC_CD")
    out = pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "MEDIAN_ALLOWED_PER_LINE": eg[config.ALLOWED].median(),
        "TOTAL_ALLOWED": eg[config.ALLOWED].sum(),
        "DENIED_OR_ZERO_LINES": g.apply(
            lambda x: ((x["PAID_DENIED"] == "DENIED") | (x[config.ALLOWED] == 0)).sum(),
            include_groups=False,
        ),
    })
    buckets = eligible.pivot_table(index="PROC_CD", columns="BUCKET", values=config.ALLOWED,
                                   aggfunc="sum", fill_value=0)
    for _, _, label in config.DAY_BUCKETS:
        out[f"ALLOWED_DAYS_{label}"] = buckets[label] if label in buckets else 0.0
    out["MEDIAN_ALLOWED_PER_LINE"] = out["MEDIAN_ALLOWED_PER_LINE"].fillna(0.0)
    out["TOTAL_ALLOWED"] = out["TOTAL_ALLOWED"].fillna(0.0)
    return out.reset_index().sort_values("CLAIM_LINES", ascending=False, ignore_index=True)


def bucket_summary(followup, window):
    """Claim lines, episodes and allowed amount per day bucket after discharge."""
    col = _day_col(followup)
    fu = in_window(followup, window).assign(BUCKET=lambda d: d[col].map(day_bucket))
    eligible = cost_eligible_lines(fu)
    g = fu.groupby("BUCKET")
    eg = eligible.groupby("BUCKET")
    return pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "TOTAL_ALLOWED": eg[config.ALLOWED].sum().reindex(g.size().index).fillna(0.0),
    }).reset_index()


def burden_stats(summary, col="POSTOP_ALLOWED"):
    s = cost_stat_series(summary, col).dropna()
    if s.empty:
        nan = float("nan")
        return {"n": 0, "median": nan, "mean": nan, "min": nan, "max": nan,
                "p25": nan, "p75": nan, "p90": nan}
    return {"n": len(s), "median": s.median(), "mean": s.mean(), "min": s.min(), "max": s.max(),
            "p25": s.quantile(0.25), "p75": s.quantile(0.75), "p90": s.quantile(0.90)}


def _group_episodes(summary, by):
    stats = summary.copy()
    for col in ["INDEX_ALLOWED", "POSTOP_ALLOWED", "TOTAL_ALLOWED"]:
        stats[col] = cost_stat_series(summary, col)
    g = stats.groupby(by, dropna=False)
    return pd.DataFrame({
        "EPISODES": g.size(),
        "MEDIAN_INDEX_ALLOWED": g["INDEX_ALLOWED"].median(),
        "MEDIAN_POSTOP_ALLOWED": g["POSTOP_ALLOWED"].median(),
        "MEDIAN_TOTAL_ALLOWED": g["TOTAL_ALLOWED"].median(),
        "MEDIAN_POSTOP_LINES": g["POSTOP_LINES"].median(),
    }).reset_index()


def by_age_group(summary):
    return _group_episodes(summary, "AGE_GROUP").sort_values("AGE_GROUP", ignore_index=True)


def by_index_diag(summary):
    return _group_episodes(summary, ["INDEX_DIAG1_CD", "INDEX_DIAG1_DESC"]).sort_values(
        "EPISODES", ascending=False, ignore_index=True)


def index_org_outcomes(summary):
    out = _group_episodes(summary, ["BILL_TIN", "BILL_PROV_NM"])
    return out.sort_values(["EPISODES", "MEDIAN_POSTOP_ALLOWED"], ascending=[False, False], ignore_index=True)


def diag_proc_combos(followup, window):
    fu = in_window(followup, window)
    eligible = cost_eligible_lines(fu)
    g = fu.groupby(["DIAG1_CD", "DIAG1_DESC", "PROC_CD", "BILL_TIN"], dropna=False)
    eg = eligible.groupby(["DIAG1_CD", "DIAG1_DESC", "PROC_CD", "BILL_TIN"], dropna=False)
    out = pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "MEDIAN_ALLOWED_PER_LINE": eg[config.ALLOWED].median(),
        "TOTAL_ALLOWED": eg[config.ALLOWED].sum(),
    })
    out["MEDIAN_ALLOWED_PER_LINE"] = out["MEDIAN_ALLOWED_PER_LINE"].fillna(0.0)
    out["TOTAL_ALLOWED"] = out["TOTAL_ALLOWED"].fillna(0.0)
    return out.reset_index().sort_values("CLAIM_LINES", ascending=False, ignore_index=True)


def billing_rendering(followup, window, bill_tin=None):
    fu = in_window(followup, window)
    if bill_tin is not None:
        fu = fu[fu["BILL_TIN"] == bill_tin]
    eligible = cost_eligible_lines(fu)
    g = fu.groupby(["BILL_TIN", "BILL_PROV_NM", "REND_NPI"], dropna=False)
    eg = eligible.groupby(["BILL_TIN", "BILL_PROV_NM", "REND_NPI"], dropna=False)
    out = pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "TOTAL_ALLOWED": eg[config.ALLOWED].sum(),
        "MEDIAN_ALLOWED_PER_LINE": eg[config.ALLOWED].median(),
    })
    out["TOTAL_ALLOWED"] = out["TOTAL_ALLOWED"].fillna(0.0)
    out["MEDIAN_ALLOWED_PER_LINE"] = out["MEDIAN_ALLOWED_PER_LINE"].fillna(0.0)
    return out.reset_index().sort_values("CLAIM_LINES", ascending=False, ignore_index=True)


def assign_cohorts(summary, low_pct=25, high_pct=75, col="POSTOP_ALLOWED"):
    s = cost_stat_series(summary, col)
    eligible = s.dropna()
    if eligible.empty:
        lo = hi = float("nan")
        cohort = pd.Series(pd.NA, index=summary.index, dtype="object")
    else:
        lo, hi = eligible.quantile(low_pct / 100), eligible.quantile(high_pct / 100)
        cohort = pd.Series(pd.NA, index=summary.index, dtype="object")
        mask = s.notna()
        cohort[mask & (s <= lo)] = "Lower"
        cohort[mask & (s > lo) & (s <= hi)] = "Typical"
        cohort[mask & (s > hi)] = "Elevated"
    return summary.assign(COHORT=pd.Categorical(cohort, categories=COHORTS, ordered=True)), lo, hi


def cohort_summary(cohorted):
    stats = cohorted.copy()
    stats["POSTOP_ALLOWED"] = cost_stat_series(cohorted, "POSTOP_ALLOWED")
    stats["INDEX_ALLOWED"] = cost_stat_series(cohorted, "INDEX_ALLOWED")
    g = stats.groupby("COHORT", observed=False)
    return pd.DataFrame({
        "EPISODES": g.size(),
        "MEDIAN_POSTOP_ALLOWED": g["POSTOP_ALLOWED"].median(),
        "MEDIAN_INDEX_ALLOWED": g["INDEX_ALLOWED"].median(),
        "MEDIAN_POSTOP_LINES": g["POSTOP_LINES"].median(),
    }).reset_index()


def flag_threshold(summary, dollars, col="POSTOP_ALLOWED"):
    s = cost_stat_series(summary, col)
    eligible = s.dropna()
    n = len(eligible)
    flagged = int((eligible > dollars).sum()) if n else 0
    return {"flagged": flagged, "total": n, "share": flagged / n if n else 0.0}


def org_cohort_counts(cohorted):
    assigned = cohorted[cohorted["COHORT"].notna()]
    t = pd.crosstab(assigned["BILL_TIN"], assigned["COHORT"]).reindex(columns=COHORTS, fill_value=0)
    t.insert(0, "EPISODES", t.sum(axis=1))
    return t.reset_index().sort_values("EPISODES", ascending=False, ignore_index=True)


def window_sensitivity(episodes, followup, sameday, low_pct=25, high_pct=75):
    from src.episodes import episode_summary

    out = episodes[["EPISODE_ID"]].copy()
    for w in config.WINDOWS:
        c, _, _ = assign_cohorts(episode_summary(episodes, followup, sameday, w), low_pct, high_pct)
        out[f"POSTOP_{w}"] = c["POSTOP_ALLOWED"].values
        out[f"COHORT_{w}"] = c["COHORT"].astype(str).values
    out["COHORT_CHANGED"] = out[[f"COHORT_{w}" for w in config.WINDOWS]].nunique(axis=1) > 1
    return out


def stay_vs_postop_split(summary):
    index_stat = cost_stat_series(summary, "INDEX_ALLOWED")
    postop_stat = cost_stat_series(summary, "POSTOP_ALLOWED")
    return pd.DataFrame({
        "Component": ["Index hospitalization", "After discharge"],
        "EPISODES_WITH_DOLLARS": [int(index_stat.notna().sum()), int(postop_stat.notna().sum())],
        "MEDIAN_ALLOWED": [index_stat.median(), postop_stat.median()],
        "TOTAL_ALLOWED": [
            summary.loc[index_stat.notna(), "INDEX_ALLOWED"].sum(),
            summary.loc[postop_stat.notna(), "POSTOP_ALLOWED"].sum(),
        ],
    })


def major_event_summary(events):
    cols = ["EVENT_TYPE", "RELATEDNESS", "EVENTS", "EPISODES", "TOTAL_ALLOWED", "MEDIAN_ALLOWED"]
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=cols)
    if "RELATEDNESS" not in events.columns:
        events = events.assign(RELATEDNESS="Unrelated")
    g = events.groupby(["EVENT_TYPE", "RELATEDNESS"], dropna=False)
    return pd.DataFrame({
        "EVENTS": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "TOTAL_ALLOWED": g["ALLOWED"].sum(),
        "MEDIAN_ALLOWED": g["ALLOWED"].median(),
    }).reset_index().sort_values(["RELATEDNESS", "TOTAL_ALLOWED"], ascending=[True, False], ignore_index=True)
