"""Descriptive aggregations over episodes and follow-up lines. No causal or quality scoring."""
import pandas as pd

from src import config

COHORTS = ["Lower", "Typical", "Elevated"]


def in_window(followup, window):
    return followup[followup["DAYS_AFTER_SURGERY"] <= window]


def day_bucket(days):
    for lo, hi, label in config.DAY_BUCKETS:
        if lo <= days <= hi:
            return label
    return ">90"


def timeline(followup, episode_id, window):
    """One episode's post-index lines in date order with a running allowed total."""
    t = in_window(followup[followup["EPISODE_ID"] == episode_id], window)
    t = t.sort_values(["SRVC_STRT_DT", "ADJD_MED_CLM_SRVC_LN_NUM"]).copy()
    t["RUNNING_ALLOWED"] = t[config.ALLOWED].cumsum()
    cols = ["SRVC_STRT_DT", "DAYS_AFTER_SURGERY", "PROC_CD", "DIAG1_CD", "DIAG1_DESC", "PLSRV_CD",
            "BILL_TIN", "BILL_PROV_NM", "REND_NPI", "PAID_DENIED", config.ALLOWED, "RUNNING_ALLOWED"]
    return t[cols].reset_index(drop=True)


def proc_code_summary(followup, window):
    """Per PROC_CD: claim lines, distinct episodes, allowed stats and allowed by day bucket."""
    fu = in_window(followup, window).assign(BUCKET=lambda d: d["DAYS_AFTER_SURGERY"].map(day_bucket))
    g = fu.groupby("PROC_CD")
    out = pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "MEDIAN_ALLOWED_PER_LINE": g[config.ALLOWED].median(),
        "TOTAL_ALLOWED": g[config.ALLOWED].sum(),
        "DENIED_OR_ZERO_LINES": g.apply(lambda x: ((x["PAID_DENIED"] == "DENIED") | (x[config.ALLOWED] == 0)).sum(),
                                        include_groups=False),
    })
    buckets = fu.pivot_table(index="PROC_CD", columns="BUCKET", values=config.ALLOWED, aggfunc="sum", fill_value=0)
    for _, _, label in config.DAY_BUCKETS:
        out[f"ALLOWED_DAYS_{label}"] = buckets[label] if label in buckets else 0.0
    return out.reset_index().sort_values("CLAIM_LINES", ascending=False, ignore_index=True)


def bucket_summary(followup, window):
    """Claim lines, episodes and allowed amount per 0-30 / 31-60 / 61-90 day bucket."""
    fu = in_window(followup, window).assign(BUCKET=lambda d: d["DAYS_AFTER_SURGERY"].map(day_bucket))
    g = fu.groupby("BUCKET")
    return pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "TOTAL_ALLOWED": g[config.ALLOWED].sum(),
    }).reset_index()


def burden_stats(summary, col="POSTOP_ALLOWED"):
    s = summary[col]
    return {"n": len(s), "median": s.median(), "mean": s.mean(), "min": s.min(), "max": s.max(),
            "p25": s.quantile(0.25), "p75": s.quantile(0.75), "p90": s.quantile(0.90)}


def _group_episodes(summary, by):
    g = summary.groupby(by, dropna=False)
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
    return _group_episodes(summary, ["INDEX_DIAG1_CD", "INDEX_DIAG1_DESC"]).sort_values("EPISODES", ascending=False, ignore_index=True)


def index_org_outcomes(summary):
    """Episodes attributed to the billing org of the index surgery (not later billers)."""
    out = _group_episodes(summary, ["BILL_TIN", "BILL_PROV_NM"])
    return out.sort_values(["EPISODES", "MEDIAN_POSTOP_ALLOWED"], ascending=[False, False], ignore_index=True)


def diag_proc_combos(followup, window):
    """Follow-up DIAG1_CD x PROC_CD x billing org, with sample sizes."""
    g = in_window(followup, window).groupby(["DIAG1_CD", "DIAG1_DESC", "PROC_CD", "BILL_TIN"], dropna=False)
    return pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "MEDIAN_ALLOWED_PER_LINE": g[config.ALLOWED].median(),
        "TOTAL_ALLOWED": g[config.ALLOWED].sum(),
    }).reset_index().sort_values("CLAIM_LINES", ascending=False, ignore_index=True)


def billing_rendering(followup, window, bill_tin=None):
    """Services billed/rendered by each BILL_TIN -> REND_NPI pair in follow-up lines."""
    fu = in_window(followup, window)
    if bill_tin is not None:
        fu = fu[fu["BILL_TIN"] == bill_tin]
    g = fu.groupby(["BILL_TIN", "BILL_PROV_NM", "REND_NPI"], dropna=False)
    return pd.DataFrame({
        "CLAIM_LINES": g.size(),
        "EPISODES": g["EPISODE_ID"].nunique(),
        "TOTAL_ALLOWED": g[config.ALLOWED].sum(),
        "MEDIAN_ALLOWED_PER_LINE": g[config.ALLOWED].median(),
    }).reset_index().sort_values("CLAIM_LINES", ascending=False, ignore_index=True)


def assign_cohorts(summary, low_pct=25, high_pct=75, col="POSTOP_ALLOWED"):
    """Lower / Typical / Elevated burden by observed percentiles of the selected episodes."""
    lo, hi = summary[col].quantile(low_pct / 100), summary[col].quantile(high_pct / 100)
    cohort = pd.Series("Typical", index=summary.index)
    cohort[summary[col] <= lo] = "Lower"
    cohort[summary[col] > hi] = "Elevated"
    return summary.assign(COHORT=pd.Categorical(cohort, categories=COHORTS, ordered=True)), lo, hi


def cohort_summary(cohorted):
    g = cohorted.groupby("COHORT", observed=False)
    return pd.DataFrame({
        "EPISODES": g.size(),
        "MEDIAN_POSTOP_ALLOWED": g["POSTOP_ALLOWED"].median(),
        "MEDIAN_INDEX_ALLOWED": g["INDEX_ALLOWED"].median(),
        "MEDIAN_POSTOP_LINES": g["POSTOP_LINES"].median(),
    }).reset_index()


def flag_threshold(summary, dollars, col="POSTOP_ALLOWED"):
    flagged = summary[col] > dollars
    n = len(summary)
    return {"flagged": int(flagged.sum()), "total": n, "share": flagged.mean() if n else 0.0}


def org_cohort_counts(cohorted):
    """Per index billing org: episode count and how many fall in each cohort."""
    t = pd.crosstab(cohorted["BILL_TIN"], cohorted["COHORT"]).reindex(columns=COHORTS, fill_value=0)
    t.insert(0, "EPISODES", t.sum(axis=1))
    return t.reset_index().sort_values("EPISODES", ascending=False, ignore_index=True)


def window_sensitivity(episodes, followup, sameday, low_pct=25, high_pct=75):
    """How each episode's cohort changes across the 30/60/90-day windows."""
    from src.episodes import episode_summary

    out = episodes[["EPISODE_ID"]].copy()
    for w in config.WINDOWS:
        c, _, _ = assign_cohorts(episode_summary(episodes, followup, sameday, w), low_pct, high_pct)
        out[f"POSTOP_{w}"] = c["POSTOP_ALLOWED"].values
        out[f"COHORT_{w}"] = c["COHORT"].astype(str).values
    out["COHORT_CHANGED"] = out[[f"COHORT_{w}" for w in config.WINDOWS]].nunique(axis=1) > 1
    return out
