"""Streamlit dashboard for the postoperative cost pilot. Native components only.

Run with:  streamlit run src/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, metrics  # noqa: E402
from src.episodes import build_major_events, episode_summary, read_outputs  # noqa: E402

# Bin edges in dollars; chart x-axis is those edges / 1000 (unit: $1,000).
COST_BINS = [-float("inf"), 0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, float("inf")]
# Upper edge of each bin in $1,000; last bin is open-ended (>$100k → ">100").
COST_AXIS_K = ["0", "1", "5", "10", "25", "50", "100", ">100"]
SMALL_SAMPLE = 5

# printf-style format works on older Streamlit; named "dollar" needs >= 1.43
MONEY = st.column_config.NumberColumn(format="$%.2f")


def money_config(df):
    return {c: MONEY for c in df.columns if "ALLOWED" in c or c == "ALLOWED"}


def show_table(df, **kwargs):
    st.dataframe(df, hide_index=True, column_config=money_config(df), **kwargs)


def highlight_relatedness(row):
    """Row colors via pandas Styler (no custom CSS). Related = green, Unrelated = gray; text black."""
    relatedness = str(row.get("RELATEDNESS", "")).strip()
    if relatedness == "Related":
        style = "background-color: #d4edda; color: #000000"
    elif relatedness == "Unrelated":
        style = "background-color: #f0f0f0; color: #000000"
    else:
        return [""] * len(row)
    return [style] * len(row)


def show_relatedness_table(df, **kwargs):
    """Drill-down event table with Related / Unrelated row highlighting."""
    if df is None or df.empty:
        show_table(df, **kwargs)
        return
    styled = df.reset_index(drop=True).style.apply(highlight_relatedness, axis=1)
    money_cols = [c for c in df.columns if "ALLOWED" in c or c == "ALLOWED"]
    if money_cols:
        styled = styled.format({c: "${:,.2f}" for c in money_cols}, na_rep="")
    st.dataframe(styled, hide_index=True, column_config=money_config(df), **kwargs)


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
st.caption(
    "Exploratory cost and utilization view. "
    "The index episode is the full hospitalization tied to CPT 63280 (or the surgery day if not inpatient). "
    "Follow-up starts after discharge. Major events are flagged with a spine/neuro family heuristic "
    "(Related / Unrelated); that is not clinical adjudication."
)

data = load()
episodes = data["index_episodes"]
followup = data["followup_claims"]
sameday = data.get("sameday_claims", followup.iloc[0:0].copy())

# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("Filters")
sb.selectbox("Index CPT", sorted(episodes["INDEX_CPT"].unique()))
window = sb.select_slider(
    "Follow-up window (days after discharge)",
    options=config.WINDOWS,
    value=config.DEFAULT_WINDOW,
)
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
exclude_denied_index = sb.checkbox("Exclude episodes with no eligible index stay dollars", value=False)

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
ev = build_major_events(fu, window, episodes=summary)

sb.caption(f"{len(summary)} of {len(episodes)} index episodes selected.")

if summary.empty:
    st.info("No episodes match the current filters. Widen the filters in the sidebar.")
    st.stop()

tab1, tab2 = st.tabs(["Overview", "Drill Down"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    st.subheader(f"Overview — CPT {config.INDEX_CPT}, {window}-day window after discharge")
    index_stat = metrics.cost_stat_series(summary, "INDEX_ALLOWED")
    postop_stat = metrics.cost_stat_series(summary, "POSTOP_ALLOWED")
    total_stat = metrics.cost_stat_series(summary, "TOTAL_ALLOWED")

    c = st.columns(4)
    c[0].metric("Index episodes", len(summary))
    c[1].metric("Median index stay allowed", fmt_money(index_stat.median()))
    c[2].metric(f"Median postop allowed ({window}d)", fmt_money(postop_stat.median()))
    c[3].metric("Median total allowed", fmt_money(total_stat.median()))

    stats = metrics.burden_stats(summary)

    def md_money(x):
        return fmt_money(x).replace("$", r"\$")

    excluded_postop = int(postop_stat.isna().sum())
    st.caption(
        "Postoperative Allowed Amount Statistics (after discharge): "
        f"P25: {md_money(stats['p25'])} | "
        f"P75: {md_money(stats['p75'])} | "
        f"Min: {md_money(stats['min'])} | "
        f"Max: {md_money(stats['max'])} | "
        f"Mean: {md_money(stats['mean'])} "
        f"({stats['n']} episodes with positive allowed; {excluded_postop} excluded from cost statistics)"
    )

    st.markdown("**Index hospitalization vs after-discharge allowed**")
    split = metrics.stay_vs_postop_split(summary).set_index("Component")[["TOTAL_ALLOWED"]]
    st.bar_chart(split)

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"**Episodes by postoperative allowed cost ({window} days after discharge)**"
        )
        st.caption("X-axis is postoperative allowed in $1,000, ascending. >100 means more than $100,000.")
        dist_source = postop_stat.dropna()
        dist = pd.cut(dist_source, COST_BINS, labels=COST_AXIS_K)
        counts = dist.value_counts().reindex(COST_AXIS_K).fillna(0).astype(int)
        cost_dist = pd.DataFrame({
            "Postop allowed ($1,000)": pd.Categorical(
                COST_AXIS_K, categories=COST_AXIS_K, ordered=True
            ),
            "Episodes": counts.to_numpy(),
        })
        st.bar_chart(cost_dist, x="Postop allowed ($1,000)", y="Episodes")
    with right:
        st.markdown("**Major follow-up events** (heuristic Related / Unrelated; not quality scores)")
        st.caption(
            "Related uses the spine/neuro family list, exact index DIAG1 match, or index CPT repeat. "
            "Not clinical adjudication."
        )
        show_relatedness_table(metrics.major_event_summary(ev))
        st.caption("Row colors: Related = light green · Unrelated = gray.")
        related_n = int(ev["RELATED_TO_INDEX"].sum()) if len(ev) and "RELATED_TO_INDEX" in ev else 0
        unrelated_n = int(len(ev) - related_n) if len(ev) else 0
        c2 = st.columns(4)
        c2[0].metric("Related events", related_n)
        c2[1].metric("Unrelated events", unrelated_n)
        c2[2].metric("Episodes with later inpatient", int((summary["SUBSEQUENT_INPATIENT_EVENTS"] > 0).sum()))
        c2[3].metric("Episodes with later OP surgery", int((summary["OUTPATIENT_SURGERY_EVENTS"] > 0).sum()))

    left, right = st.columns(2)
    with left:
        st.markdown("**By age group** (episode counts; age is context, not risk adjustment)")
        show_table(metrics.by_age_group(summary))
    with right:
        st.markdown("**By Billing Organization** (episodes attributed to the surgery's billing organization)")
        show_table(metrics.index_org_outcomes(summary))

    st.markdown("**By index diagnosis**")
    show_table(metrics.by_index_diag(summary))

# ---------------- Tab 2: Drill Down ----------------
with tab2:
    st.subheader("Episode timeline")
    drill_index = metrics.cost_stat_series(summary, "INDEX_ALLOWED")
    drill_postop = metrics.cost_stat_series(summary, "POSTOP_ALLOWED")
    drill_total = metrics.cost_stat_series(summary, "TOTAL_ALLOWED")
    labels = {
        r.EPISODE_ID: (
            f"{r.EPISODE_ID} — surgery {r.INDEX_SRVC_DT:%Y-%m-%d}, "
            f"discharge {pd.Timestamp(r.STAY_END):%Y-%m-%d}, "
            f"postop {fmt_money(drill_postop.loc[i])}"
        )
        for i, r in summary.iterrows()
    }
    ep_id = st.selectbox("Episode", list(labels), format_func=labels.get)
    ep_row = summary["EPISODE_ID"] == ep_id
    ep = summary[ep_row].iloc[0]
    ep_idx = summary.index[ep_row][0]

    c = st.columns(4)
    c[0].metric("Index stay allowed", fmt_money(drill_index.loc[ep_idx]))
    c[1].metric(f"Postop allowed ({window}d)", fmt_money(drill_postop.loc[ep_idx]))
    c[2].metric("Total allowed", fmt_money(drill_total.loc[ep_idx]))
    c[3].metric("Postop claim lines", int(ep["POSTOP_LINES"]))
    stay_label = "inpatient" if bool(ep.get("INPATIENT", True)) else "surgery-day (not inpatient)"
    st.caption(
        f"Billing Organization {ep['BILL_PROV_NM']} · index surgeon NPI {ep['INDEX_REND_NPI']} · "
        f"index diagnosis {ep['INDEX_DIAG1_CD']} {ep['INDEX_DIAG1_DESC']} · "
        f"stay {pd.Timestamp(ep['STAY_START']):%Y-%m-%d} to {pd.Timestamp(ep['STAY_END']):%Y-%m-%d} "
        f"({stay_label}, LOS {int(ep['LOS_DAYS'])} days)"
    )

    ep_events = ev[ev["EPISODE_ID"] == ep_id] if len(ev) else ev
    st.markdown("**Major follow-up events after discharge**")
    st.caption(
        "Related / Unrelated is a spine/neuro family heuristic for investigation, "
        "not definitive evidence of surgical quality or clinical relatedness. "
        "Row colors: Related = light green · Unrelated = gray."
    )
    relatedness_filter = st.selectbox("Event relatedness", ["All", "Related", "Unrelated"])
    if ep_events is None or ep_events.empty:
        st.info("No subsequent inpatient or outpatient-surgery events in this window.")
    else:
        shown = ep_events
        if relatedness_filter != "All" and "RELATEDNESS" in shown.columns:
            shown = shown[shown["RELATEDNESS"] == relatedness_filter]
        if shown.empty:
            st.info(f"No {relatedness_filter.lower()} major events for this episode in this window.")
        else:
            show_cols = [
                "EVENT_TYPE", "START_DT", "END_DT", "RELATEDNESS", "RELATEDNESS_REASON",
                "LINES", "ALLOWED", "TOP_PROC_CD", "TOP_DIAG1_CD", "TOP_DIAG1_DESC",
            ]
            show_relatedness_table(shown[[c for c in show_cols if c in shown.columns]])

    tl = metrics.timeline(followup, ep_id, window)
    if tl.empty:
        st.info("No post-discharge claim lines in this window.")
    else:
        day_col = "DAYS_AFTER_DISCHARGE" if "DAYS_AFTER_DISCHARGE" in tl.columns else "DAYS_AFTER_SURGERY"
        daily = tl.groupby(day_col)["RUNNING_ALLOWED"].last().reset_index()
        daily = daily.rename(columns={
            day_col: "Days after discharge",
            "RUNNING_ALLOWED": "Running postop allowed ($)",
        })
        st.line_chart(daily, x="Days after discharge", y="Running postop allowed ($)")
        show_table(tl)

    st.markdown("---")
    st.subheader("PostOperative providers")
    org_names = pd.concat([summary[["BILL_TIN", "BILL_PROV_NM"]], fu[["BILL_TIN", "BILL_PROV_NM"]]]) \
                  .dropna().drop_duplicates("BILL_TIN").set_index("BILL_TIN")["BILL_PROV_NM"]
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
        cols = [
            "EPISODE_ID", "INDEX_SRVC_DT", "STAY_END", "INDEX_REND_NPI", "INDEX_ALLOWED", "POSTOP_ALLOWED",
            "TOTAL_ALLOWED", "POSTOP_LINES", "SUBSEQUENT_INPATIENT_EVENTS", "OUTPATIENT_SURGERY_EVENTS",
            "RELATED_MAJOR_EVENTS", "UNRELATED_MAJOR_EVENTS",
        ]
        show_table(org_eps[[c for c in cols if c in org_eps.columns]])
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
        f"- **Index episode:** CPT {config.INDEX_CPT} anchors the case. When POS {config.INPATIENT_POS} is present "
        "on the surgery date, the index is the contiguous inpatient stay (all claim lines from stay start through "
        "discharge). Otherwise the index is the surgery-day claims.\n"
        f"- **Postoperative:** claims with a service date after discharge, within the {window}-day window, and a "
        "2026 process date.\n"
        "- **Major events:** later POS 21 runs (subsequent inpatient) and POS 22/24 days (outpatient surgery/"
        "facility) are flagged for review. **Related / Unrelated** uses a deterministic spine/neuro DIAG1 prefix "
        "and CPT family list plus exact index DIAG1 / index CPT match. This is not clinical adjudication.\n"
        "- **Cost statistics** omit denied claims and claims with zero allowed amount. Those claims are still "
        "counted separately."
    )
