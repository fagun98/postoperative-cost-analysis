"""Streamlit dashboard for the postoperative cost pilot. Native components only.

Run with:  streamlit run src/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, metrics  # noqa: E402
from src.episodes import episode_summary, read_outputs  # noqa: E402

COST_BINS = [-float("inf"), 0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, float("inf")]
COST_LABELS = ["$0 or less", "$0-1k", "$1k-5k", "$5k-10k", "$10k-25k", "$25k-50k", "$50k-100k", "$100k+"]
SMALL_SAMPLE = 5

# printf-style format works on older Streamlit; named "dollar" needs >= 1.43
MONEY = st.column_config.NumberColumn(format="$%.2f")


def money_config(df):
    return {c: MONEY for c in df.columns if "ALLOWED" in c}


def show_table(df, **kwargs):
    st.dataframe(df, hide_index=True, column_config=money_config(df), **kwargs)


def fmt_money(x):
    return f"${x:,.0f}" if pd.notna(x) else "n/a"


def org_label(tin, names, index_orgs=None):
    name = names.get(tin, "")
    label = name if pd.notna(name) and str(name).strip() else str(tin)
    if index_orgs is not None and tin in index_orgs:
        label = f"{label} (index surgery)"
    return label


@st.cache_data
def load():
    return read_outputs()


st.set_page_config(page_title="Postoperative Cost", layout="wide")
st.title("Postoperative Cost Analysis")
st.caption("Exploratory cost and utilization view. "
           "Follow-up currently includes all postoperative claim within the observation window, whether or not it is clinically related to the surgery.")

data = load()
episodes, followup, sameday = data["index_episodes"], data["followup_claims"], data["sameday_claims"]

# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("Filters")
sb.selectbox("Index CPT", sorted(episodes["INDEX_CPT"].unique()))
window = sb.select_slider("Follow-up window (days after surgery)", options=config.WINDOWS, value=config.DEFAULT_WINDOW)
ages = sb.multiselect("Age group", sorted(episodes["AGE_GROUP"].unique()))
diags = sb.multiselect("Index diagnosis (DIAG1)", sorted(episodes["INDEX_DIAG1_CD"].dropna().unique()))

org_name_lookup = (
    episodes[["BILL_TIN", "BILL_PROV_NM"]].dropna(subset=["BILL_TIN"])
    .drop_duplicates("BILL_TIN").set_index("BILL_TIN")["BILL_PROV_NM"]
)
org_choices = sorted(episodes["BILL_TIN"].dropna().unique(), key=lambda t: org_label(t, org_name_lookup).lower())
orgs = sb.multiselect(
    "Billing Organization",
    org_choices,
    format_func=lambda t: org_label(t, org_name_lookup),
)
complete_only = sb.checkbox(f"Only episodes with full {window}-day observation", value=False)
exclude_conflict = sb.checkbox("Exclude episodes with identity conflicts", value=False)
exclude_denied_index = sb.checkbox("Exclude episodes whose index lines were all denied", value=False)
threshold = sb.number_input("Postoperative allowed-cost flag ($)", min_value=0, value=25_000, step=1_000)

summary = episode_summary(episodes, followup, sameday, window)
mask = pd.Series(True, index=summary.index)
if ages:
    mask &= summary["AGE_GROUP"].isin(ages)
if diags:
    mask &= summary["INDEX_DIAG1_CD"].isin(diags)
if orgs:
    mask &= summary["BILL_TIN"].isin(orgs)
if complete_only:
    mask &= summary["COMPLETE_FOLLOWUP"]
if exclude_conflict:
    mask &= ~summary["IDENTITY_CONFLICT"]
if exclude_denied_index:
    mask &= ~summary["INDEX_ALL_DENIED"]
summary = summary[mask].reset_index(drop=True)
fu = followup[followup["EPISODE_ID"].isin(summary["EPISODE_ID"])]

sb.caption(f"{len(summary)} of {len(episodes)} index episodes selected.")

if summary.empty:
    st.info("No episodes match the current filters. Widen the filters in the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "PostOperative Care Patterns", "Cohorts & Thresholds", "Drill Down"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    st.subheader(f"Overview — CPT {config.INDEX_CPT}, {window}-day window")
    index_stat = metrics.cost_stat_series(summary, "INDEX_ALLOWED")
    postop_stat = metrics.cost_stat_series(summary, "POSTOP_ALLOWED")
    total_stat = metrics.cost_stat_series(summary, "TOTAL_ALLOWED")

    c = st.columns(4)
    c[0].metric("Index episodes", len(summary))
    c[1].metric("Median index allowed", fmt_money(index_stat.median()))
    c[2].metric(f"Median postop allowed ({window}d)", fmt_money(postop_stat.median()))
    c[3].metric("Median total allowed", fmt_money(total_stat.median()))

    stats = metrics.burden_stats(summary)

    def md_money(x):
        return fmt_money(x).replace("$", r"\$")

    excluded_postop = int(postop_stat.isna().sum())
    st.caption(
        "Postoperative Allowed Amount Statistics: "
        f"P25: {md_money(stats['p25'])} | "
        f"P75: {md_money(stats['p75'])} | "
        f"Min: {md_money(stats['min'])} | "
        f"Max: {md_money(stats['max'])} | "
        f"Mean: {md_money(stats['mean'])} "
        f"({stats['n']} episodes with positive allowed; {excluded_postop} excluded from cost statistics)"
    )
    st.markdown(f"**Episodes by postoperative allowed cost ({window} days)**")
    dist_source = postop_stat.dropna()
    dist = pd.cut(dist_source, COST_BINS, labels=COST_LABELS).value_counts().reindex(COST_LABELS)
    # Index order = bar order (compatible with Streamlit < 1.50, which has no sort=)
    cost_dist = dist.rename("Episodes")
    cost_dist.index = pd.Index(COST_LABELS, name="Postop allowed")
    st.bar_chart(cost_dist)

    left, right = st.columns(2)
    with left:
        st.markdown("**By age group** (episode counts; age is context, not risk adjustment)")
        show_table(metrics.by_age_group(summary))
    with right:
        st.markdown("**By Billing Organization** (episodes attributed to the surgery's billing organization)")
        show_table(metrics.index_org_outcomes(summary))

    st.markdown("**By index diagnosis**")
    show_table(metrics.by_index_diag(summary))

# ---------------- Tab 2: Care Patterns ----------------
with tab2:
    st.subheader(f"Follow-up services within {window} days")
    if fu.empty:
        st.info("No follow-up claim lines for the selected episodes in this window.")
    else:
        procs = metrics.proc_code_summary(fu, window)
        left, right = st.columns(2)
        with left:
            st.markdown("**Top 10 procedure codes by claim lines**")
            by_lines = procs.head(10).set_index("PROC_CD")[["CLAIM_LINES"]]
            st.bar_chart(by_lines)
        with right:
            st.markdown("**Top 10 procedure codes by total allowed**")
            by_allowed = procs.nlargest(10, "TOTAL_ALLOWED").set_index("PROC_CD")[["TOTAL_ALLOWED"]]
            st.bar_chart(by_allowed)

        st.markdown("**Timing: claim lines and allowed by days after surgery**")
        show_table(metrics.bucket_summary(fu, window))

        st.markdown("**All follow-up procedure codes** (claim lines, distinct episodes, allowed per line)")
        show_table(procs)

        with st.expander("Follow-up diagnosis × procedure × Billing Organization"):
            st.caption("Follow-up diagnoses can differ from the index diagnosis.")
            show_table(metrics.diag_proc_combos(fu, window))

        st.markdown("**Index allowed vs. postoperative allowed** (association only)")
        scatter = summary.assign(
            **{
                "Index allowed ($)": index_stat,
                f"Postop allowed, {window}d ($)": postop_stat,
            }
        ).dropna(subset=["Index allowed ($)", f"Postop allowed, {window}d ($)"])
        st.scatter_chart(scatter, x="Index allowed ($)", y=f"Postop allowed, {window}d ($)")

# ---------------- Tab 3: Cohorts & Thresholds ----------------
with tab3:
    st.subheader("Postoperative cost-burden cohorts")
    st.caption("Cohorts come from the selected episodes' own distribution. They are investigation flags, "
               "not clinically validated cutoffs.")
    low_pct, high_pct = st.slider("Lower / Elevated percentile cutoffs", 5, 95, (25, 75), step=5)
    cohorted, lo, hi = metrics.assign_cohorts(summary, low_pct, high_pct)
    st.caption(f"Lower: ≤ {fmt_money(lo)} · Typical: {fmt_money(lo)}–{fmt_money(hi)} · Elevated: > {fmt_money(hi)}")

    cs = metrics.cohort_summary(cohorted)
    c = st.columns(3)
    for col, row in zip(c, cs.itertuples()):
        col.metric(f"{row.COHORT} burden", f"{row.EPISODES} episodes",
                   help=f"Median postop allowed {fmt_money(row.MEDIAN_POSTOP_ALLOWED)}")
    show_table(cs)

    st.markdown("---")
    st.markdown(f"**Dollar threshold: postop allowed > {fmt_money(threshold)}**")
    flag = metrics.flag_threshold(summary, threshold)
    c = st.columns(2)
    c[0].metric("Episodes above threshold", f"{flag['flagged']} of {flag['total']}")
    c[1].metric("Share above threshold", f"{flag['share']:.0%}")
    above = pd.DataFrame({
        "Episodes": [flag["total"] - flag["flagged"], flag["flagged"]],
    }, index=pd.Index(["At or below", "Above"], name="Group"))
    st.bar_chart(above)

    st.markdown("---")
    st.markdown("**Billing Organizations by cohort**")
    st.caption("Small sample and most organizations only have one episode.")
    show_table(metrics.org_cohort_counts(cohorted))

    with st.expander("Sensitivity: cohort changes across 30 / 60 / 90 days"):
        sel_eps = episodes[episodes["EPISODE_ID"].isin(summary["EPISODE_ID"])]
        sens = metrics.window_sensitivity(sel_eps, followup, sameday, low_pct, high_pct)
        st.caption(f"{int(sens['COHORT_CHANGED'].sum())} of {len(sens)} episodes change cohort between windows.")
        show_table(sens)

# ---------------- Tab 4: Drill Down ----------------
with tab4:
    st.subheader("Episode timeline")
    drill_index = metrics.cost_stat_series(summary, "INDEX_ALLOWED")
    drill_postop = metrics.cost_stat_series(summary, "POSTOP_ALLOWED")
    drill_total = metrics.cost_stat_series(summary, "TOTAL_ALLOWED")
    labels = {
        r.EPISODE_ID: (
            f"{r.EPISODE_ID} — index {r.INDEX_SRVC_DT:%Y-%m-%d}, {r.AGE_GROUP}, "
            f"postop {fmt_money(drill_postop.loc[i])}"
        )
        for i, r in summary.iterrows()
    }
    ep_id = st.selectbox("Episode", list(labels), format_func=labels.get)
    ep_row = summary["EPISODE_ID"] == ep_id
    ep = summary[ep_row].iloc[0]
    ep_idx = summary.index[ep_row][0]

    c = st.columns(4)
    c[0].metric("Index allowed", fmt_money(drill_index.loc[ep_idx]))
    c[1].metric(f"Postop allowed ({window}d)", fmt_money(drill_postop.loc[ep_idx]))
    c[2].metric("Total allowed", fmt_money(drill_total.loc[ep_idx]))
    c[3].metric("Postop claim lines", int(ep["POSTOP_LINES"]))
    st.caption(f"Billing Organization {ep['BILL_PROV_NM']} · index surgeon NPI {ep['INDEX_REND_NPI']} · "
               f"index diagnosis {ep['INDEX_DIAG1_CD']} {ep['INDEX_DIAG1_DESC']} · "
               f"same-day lines {int(ep['SAMEDAY_LINES'])} ({fmt_money(ep['SAMEDAY_ALLOWED'])}, not in total)")

    tl = metrics.timeline(followup, ep_id, window)
    if tl.empty:
        st.info("No post-index claim lines in this window.")
    else:
        daily = tl.groupby("DAYS_AFTER_SURGERY")["RUNNING_ALLOWED"].last().reset_index()
        daily = daily.rename(columns={
            "DAYS_AFTER_SURGERY": "Days after surgery",
            "RUNNING_ALLOWED": "Running postop allowed ($)",
        })
        st.line_chart(daily, x="Days after surgery", y="Running postop allowed ($)")
        show_table(tl)

    st.markdown("---")
    st.subheader("PostOperative providers")
    org_names = pd.concat([summary[["BILL_TIN", "BILL_PROV_NM"]], fu[["BILL_TIN", "BILL_PROV_NM"]]]) \
                  .dropna().drop_duplicates("BILL_TIN").set_index("BILL_TIN")["BILL_PROV_NM"]
    # Index-surgery orgs first, then orgs that only billed follow-up care.
    index_orgs = sorted(summary["BILL_TIN"].dropna().unique())
    org_options = index_orgs + sorted(set(org_names.index) - set(index_orgs))
    org = st.selectbox(
        "Billing Organization",
        org_options,
        format_func=lambda t: org_label(t, org_names, index_orgs),
    )

    org_eps = summary[summary["BILL_TIN"] == org].copy()
    st.markdown(f"**Index surgeries billed by this organization: {len(org_eps)}**")
    if len(org_eps):
        org_eps["INDEX_ALLOWED"] = metrics.cost_stat_series(org_eps, "INDEX_ALLOWED")
        org_eps["POSTOP_ALLOWED"] = metrics.cost_stat_series(org_eps, "POSTOP_ALLOWED")
        org_eps["TOTAL_ALLOWED"] = metrics.cost_stat_series(org_eps, "TOTAL_ALLOWED")
        show_table(org_eps[["EPISODE_ID", "INDEX_SRVC_DT", "INDEX_REND_NPI", "INDEX_ALLOWED", "POSTOP_ALLOWED",
                            "TOTAL_ALLOWED", "POSTOP_LINES"]])
        if len(org_eps) < SMALL_SAMPLE:
            st.caption(f"Fewer than {SMALL_SAMPLE} episodes.")

    st.markdown("**Follow-up services this organization billed, by servicing NPI** "
                "(these are services the organization billed, not outcomes of its surgeries)")
    br = metrics.billing_rendering(fu, window, org)
    if br.empty:
        st.info("This organization billed no follow-up lines for the selected episodes in this window.")
    else:
        show_table(br)

# ---------------- Notes ----------------
with st.expander("Notes"):
    st.markdown(
        f"- **Episodes:** CPT {config.INDEX_CPT} INDEX claims with an index date between {config.STUDY_START:%Y-%m-%d} "
        f"and {config.STUDY_END:%Y-%m-%d}, grouped by member id + index date.\n"
        "- **Postoperative:** claims with a service date after the index date and within the window, and a 2026 process "
        "date.\n"
        "- **Cost statistics** (mean, median, percentiles, cost distribution, cohorts, and thresholds) omit denied "
        "claims and claims with zero allowed amount. Those claims are still counted separately."
    )
