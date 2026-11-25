"""Build index episodes, attach later claim lines, and summarise each episode.

The index episode is the full inpatient stay tied to CPT 63280 (POS 21), not only
the surgery-date CPT lines. Postoperative windows start the day after discharge.
"""
import pandas as pd

from src import config

EPISODE_KEY = ["PERSON_KEY", "INDEX_SRVC_DT"]


def _mode(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else pd.NA


def _norm_dates(s):
    return pd.to_datetime(s).dt.normalize()


def contiguous_stay_bounds(pos21_dates, surgery_dt):
    """Return (stay_start, stay_end) for an unbroken POS 21 run through surgery_dt.

    If surgery_dt has no POS 21 claim, there is no inpatient stay: return None.
    """
    surgery_dt = pd.Timestamp(surgery_dt).normalize()
    dates = {pd.Timestamp(d).normalize() for d in pos21_dates}
    if surgery_dt not in dates:
        return None
    start = surgery_dt
    while (start - pd.Timedelta(days=1)) in dates:
        start = start - pd.Timedelta(days=1)
    end = surgery_dt
    while (end + pd.Timedelta(days=1)) in dates:
        end = end + pd.Timedelta(days=1)
    return start, end


def _episode_claims(claims, person, index_dt):
    return claims[(claims["PERSON_KEY"] == person) & (claims["INDEX_SRVC_DT"] == index_dt)].copy()


def build_index_episodes(claims, index_cpt=config.INDEX_CPT):
    """One row per (id+DOB, index date) with stay bounds and stay-based index dollars."""
    idx = claims[
        (claims["JOURNEY_PHASE"] == "INDEX")
        & (claims["PROC_CD"] == index_cpt)
        & claims["INDEX_SRVC_DT"].between(config.STUDY_START, config.STUDY_END)
    ]
    data_cutoff = claims["PROC_DT"].max()

    rows = []
    for (person, index_dt), g in idx.groupby(EPISODE_KEY, sort=True):
        index_dt = pd.Timestamp(index_dt).normalize()
        # The surgeon is taken as the rendering NPI on the highest-allowed index CPT line.
        top = g.sort_values([config.ALLOWED, "rec"], ascending=[False, True]).iloc[0]
        all_lines = _episode_claims(claims, person, index_dt)
        all_lines["SRVC_STRT_DT"] = _norm_dates(all_lines["SRVC_STRT_DT"])

        pos21_dates = all_lines.loc[
            all_lines["PLSRV_CD"].astype(str) == config.INPATIENT_POS, "SRVC_STRT_DT"
        ]
        bounds = contiguous_stay_bounds(pos21_dates, index_dt)
        if bounds is None:
            stay_start = stay_end = index_dt
            inpatient = False
        else:
            stay_start, stay_end = bounds
            inpatient = True

        stay = all_lines[all_lines["SRVC_STRT_DT"].between(stay_start, stay_end)]
        paid = (stay["PAID_DENIED"] == "PAID") & (stay[config.ALLOWED] > 0)
        rows.append({
            "PERSON_KEY": person,
            "id": g["id"].iloc[0],
            "INDEX_SRVC_DT": index_dt,
            "STAY_START": stay_start,
            "STAY_END": stay_end,
            "INPATIENT": inpatient,
            "LOS_DAYS": int((stay_end - stay_start).days),
            "INDEX_CPT": index_cpt,
            "AGE_IN_YEARS": g["AGE_IN_YEARS"].iloc[0],
            "AGE_GROUP": g["AGE_GROUP"].iloc[0],
            "BILL_TIN": _mode(g["BILL_TIN"]),
            "BILL_PROV_NM": _mode(g["BILL_PROV_NM"]),
            "INDEX_REND_NPI": top["REND_NPI"],
            "INDEX_REND_NPIS": ";".join(sorted(g["REND_NPI"].dropna().unique())),
            "INDEX_DIAG1_CD": _mode(g["DIAG1_CD"]),
            "INDEX_DIAG1_DESC": _mode(g["DIAG1_DESC"]),
            "INDEX_LINES": len(stay),
            "INDEX_ALLOWED": stay[config.ALLOWED].sum(),
            "INDEX_PAID_LINES": int(paid.sum()),
            "INDEX_ALL_DENIED": bool(paid.sum() == 0),
            "INDEX_CPT_LINES": len(g),
            "INDEX_CPT_ALLOWED": g[config.ALLOWED].sum(),
            "INDEX_MULTI_TIN": g["BILL_TIN"].nunique() > 1,
            "IDENTITY_CONFLICT": bool(g["IDENTITY_CONFLICT"].any() | all_lines["IDENTITY_CONFLICT"].any()),
            "OBS_DAYS": (data_cutoff - stay_end).days,
        })
    ep = pd.DataFrame(rows)
    ep.insert(0, "EPISODE_ID", [f"E{i:03d}" for i in range(1, len(ep) + 1)])
    for w in config.WINDOWS:
        ep[f"COMPLETE_{w}"] = ep["OBS_DAYS"] >= w
    return ep


def attach_followup(claims, episodes):
    """Split claims into in-stay index lines, post-discharge follow-up, and excluded."""
    ep_cols = ["EPISODE_ID", "STAY_START", "STAY_END"] + EPISODE_KEY
    lines = claims.merge(episodes[ep_cols], on=EPISODE_KEY, how="inner").copy()
    lines["SRVC_STRT_DT"] = _norm_dates(lines["SRVC_STRT_DT"])
    lines["STAY_START"] = pd.to_datetime(lines["STAY_START"]).dt.normalize()
    lines["STAY_END"] = pd.to_datetime(lines["STAY_END"]).dt.normalize()
    lines["DAYS_AFTER_DISCHARGE"] = (lines["SRVC_STRT_DT"] - lines["STAY_END"]).dt.days
    # Keep surgery-relative days for audit; windowing uses discharge.
    lines["DAYS_AFTER_SURGERY"] = (lines["SRVC_STRT_DT"] - lines["INDEX_SRVC_DT"]).dt.days

    in_2026 = lines["PROC_DT"].dt.year == 2026
    in_stay = lines["SRVC_STRT_DT"].between(lines["STAY_START"], lines["STAY_END"])
    after = lines["DAYS_AFTER_DISCHARGE"] > 0

    index_stay = lines[in_stay].sort_values(
        ["EPISODE_ID", "SRVC_STRT_DT", "ADJD_MED_CLM_SRVC_LN_NUM", "rec"])
    followup = lines[in_2026 & after].sort_values(
        ["EPISODE_ID", "SRVC_STRT_DT", "ADJD_MED_CLM_SRVC_LN_NUM", "rec"])
    # Same-day relative to discharge is already inside the stay / index.
    sameday = lines.iloc[0:0].copy()
    excluded = lines[~in_stay & ~after]
    return (
        followup.reset_index(drop=True),
        sameday.reset_index(drop=True),
        excluded.reset_index(drop=True),
        index_stay.reset_index(drop=True),
    )


def _contiguous_runs(dates):
    """Yield (start, end) for each unbroken calendar run in a sorted unique date set."""
    if not dates:
        return
    ordered = sorted(pd.Timestamp(d).normalize() for d in dates)
    start = prev = ordered[0]
    for d in ordered[1:]:
        if (d - prev).days == 1:
            prev = d
            continue
        yield start, prev
        start = prev = d
    yield start, prev


def _relatedness_for_row(proc_cd, diag_cd, index_cpt, index_diag):
    """Return (related_bool, reason). Deterministic spine/neuro family heuristic."""
    proc = "" if pd.isna(proc_cd) else str(proc_cd).strip()
    diag = "" if pd.isna(diag_cd) else str(diag_cd).strip().upper()
    idx_cpt = "" if pd.isna(index_cpt) else str(index_cpt).strip()
    idx_diag = "" if pd.isna(index_diag) else str(index_diag).strip().upper()

    if proc and idx_cpt and proc == idx_cpt:
        return True, "index_cpt"
    if proc and proc in config.RELATED_PROC_CODES:
        return True, f"related_proc:{proc}"
    if diag and idx_diag and diag == idx_diag:
        return True, "exact_index_diag"
    if diag and diag in config.RELATED_DIAG_CODES:
        return True, f"related_diag:{diag}"
    if diag:
        for prefix in sorted(config.RELATED_DIAG_PREFIXES, key=len, reverse=True):
            if diag.startswith(prefix):
                return True, f"diag_prefix:{prefix}"
    return False, "outside_spine_neuro_family"


def label_event_relatedness(events, episodes):
    """Attach RELATED_TO_INDEX / RELATEDNESS / RELATEDNESS_REASON using index DIAG1+CPT."""
    cols = ["RELATED_TO_INDEX", "RELATEDNESS", "RELATEDNESS_REASON"]
    if events is None or len(events) == 0:
        empty = events.copy() if events is not None else pd.DataFrame()
        for c in cols:
            empty[c] = pd.Series(dtype="object" if c != "RELATED_TO_INDEX" else "boolean")
        return empty

    ep = episodes[["EPISODE_ID", "INDEX_CPT", "INDEX_DIAG1_CD"]].drop_duplicates("EPISODE_ID")
    out = events.merge(ep, on="EPISODE_ID", how="left")
    flags, reasons = [], []
    for row in out.itertuples(index=False):
        related, reason = _relatedness_for_row(
            row.TOP_PROC_CD, row.TOP_DIAG1_CD, row.INDEX_CPT, row.INDEX_DIAG1_CD)
        flags.append(related)
        reasons.append(reason)
    out["RELATED_TO_INDEX"] = flags
    out["RELATEDNESS"] = ["Related" if r else "Unrelated" for r in flags]
    out["RELATEDNESS_REASON"] = reasons
    return out.drop(columns=["INDEX_CPT", "INDEX_DIAG1_CD"]).reset_index(drop=True)


def build_major_events(followup, window=config.DEFAULT_WINDOW, episodes=None):
    """Subsequent inpatient stays and outpatient surgery/facility events after discharge."""
    empty_cols = [
        "EPISODE_ID", "EVENT_TYPE", "START_DT", "END_DT", "LINES", "ALLOWED",
        "TOP_PROC_CD", "TOP_DIAG1_CD", "TOP_DIAG1_DESC",
        "RELATED_TO_INDEX", "RELATEDNESS", "RELATEDNESS_REASON",
    ]
    if followup.empty:
        return pd.DataFrame(columns=empty_cols)

    fu = followup[followup["DAYS_AFTER_DISCHARGE"] <= window].copy()
    fu["SRVC_STRT_DT"] = _norm_dates(fu["SRVC_STRT_DT"])
    rows = []

    for ep_id, g in fu.groupby("EPISODE_ID"):
        inp_dates = set(g.loc[g["PLSRV_CD"].astype(str) == config.INPATIENT_POS, "SRVC_STRT_DT"])
        for start, end in _contiguous_runs(inp_dates):
            block = g[g["SRVC_STRT_DT"].between(start, end)
                      & (g["PLSRV_CD"].astype(str) == config.INPATIENT_POS)]
            top = block.sort_values(config.ALLOWED, ascending=False).iloc[0]
            rows.append({
                "EPISODE_ID": ep_id,
                "EVENT_TYPE": "subsequent_inpatient",
                "START_DT": start,
                "END_DT": end,
                "LINES": len(block),
                "ALLOWED": block[config.ALLOWED].sum(),
                "TOP_PROC_CD": top["PROC_CD"],
                "TOP_DIAG1_CD": top["DIAG1_CD"],
                "TOP_DIAG1_DESC": top["DIAG1_DESC"],
            })

        ops = g[g["PLSRV_CD"].astype(str).isin(config.OUTPATIENT_SURGERY_POS)]
        for day, block in ops.groupby("SRVC_STRT_DT"):
            top = block.sort_values(config.ALLOWED, ascending=False).iloc[0]
            rows.append({
                "EPISODE_ID": ep_id,
                "EVENT_TYPE": "outpatient_surgery",
                "START_DT": day,
                "END_DT": day,
                "LINES": len(block),
                "ALLOWED": block[config.ALLOWED].sum(),
                "TOP_PROC_CD": top["PROC_CD"],
                "TOP_DIAG1_CD": top["DIAG1_CD"],
                "TOP_DIAG1_DESC": top["DIAG1_DESC"],
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=empty_cols)
    out = out.sort_values(["EPISODE_ID", "START_DT", "EVENT_TYPE"]).reset_index(drop=True)
    if episodes is not None and len(episodes):
        out = label_event_relatedness(out, episodes)
    else:
        out["RELATED_TO_INDEX"] = False
        out["RELATEDNESS"] = "Unrelated"
        out["RELATEDNESS_REASON"] = "missing_index_context"
    return out


def episode_summary(episodes, followup, sameday, window=config.DEFAULT_WINDOW):
    """Per-episode index (stay), postoperative (days 1..window after discharge), and totals."""
    day_col = "DAYS_AFTER_DISCHARGE" if "DAYS_AFTER_DISCHARGE" in followup.columns else "DAYS_AFTER_SURGERY"
    fu = followup[followup[day_col] <= window] if len(followup) else followup
    if len(fu):
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
    else:
        agg = pd.DataFrame(columns=[
            "POSTOP_ALLOWED", "POSTOP_LINES", "POSTOP_PAID_LINES", "POSTOP_DENIED_OR_ZERO_LINES",
            "POSTOP_DISTINCT_PROC", "POSTOP_SERVICE_DAYS",
        ])

    if len(sameday):
        same = sameday.groupby("EPISODE_ID")[config.ALLOWED].agg(SAMEDAY_ALLOWED="sum", SAMEDAY_LINES="size")
    else:
        same = pd.DataFrame(columns=["SAMEDAY_ALLOWED", "SAMEDAY_LINES"])

    events = build_major_events(followup, window, episodes=episodes)
    if len(events):
        ev = events.groupby("EPISODE_ID").agg(
            MAJOR_EVENTS=("EVENT_TYPE", "size"),
            SUBSEQUENT_INPATIENT_EVENTS=("EVENT_TYPE", lambda s: int((s == "subsequent_inpatient").sum())),
            OUTPATIENT_SURGERY_EVENTS=("EVENT_TYPE", lambda s: int((s == "outpatient_surgery").sum())),
            RELATED_MAJOR_EVENTS=("RELATED_TO_INDEX", lambda s: int(s.sum())),
            UNRELATED_MAJOR_EVENTS=("RELATED_TO_INDEX", lambda s: int((~s.astype(bool)).sum())),
            MAJOR_EVENT_ALLOWED=("ALLOWED", "sum"),
        )
    else:
        ev = pd.DataFrame(columns=[
            "MAJOR_EVENTS", "SUBSEQUENT_INPATIENT_EVENTS", "OUTPATIENT_SURGERY_EVENTS",
            "RELATED_MAJOR_EVENTS", "UNRELATED_MAJOR_EVENTS", "MAJOR_EVENT_ALLOWED",
        ])

    out = episodes.merge(agg, left_on="EPISODE_ID", right_index=True, how="left") \
                  .merge(same, left_on="EPISODE_ID", right_index=True, how="left") \
                  .merge(ev, left_on="EPISODE_ID", right_index=True, how="left")
    count_cols = ["POSTOP_LINES", "POSTOP_PAID_LINES", "POSTOP_DENIED_OR_ZERO_LINES",
                  "POSTOP_DISTINCT_PROC", "POSTOP_SERVICE_DAYS", "SAMEDAY_LINES",
                  "MAJOR_EVENTS", "SUBSEQUENT_INPATIENT_EVENTS", "OUTPATIENT_SURGERY_EVENTS",
                  "RELATED_MAJOR_EVENTS", "UNRELATED_MAJOR_EVENTS"]
    for c in count_cols:
        if c not in out:
            out[c] = 0
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out[["POSTOP_ALLOWED", "SAMEDAY_ALLOWED", "MAJOR_EVENT_ALLOWED"]] = (
        out[["POSTOP_ALLOWED", "SAMEDAY_ALLOWED", "MAJOR_EVENT_ALLOWED"]].fillna(0.0))
    out["TOTAL_ALLOWED"] = out["INDEX_ALLOWED"] + out["POSTOP_ALLOWED"]
    out["WINDOW_DAYS"] = window
    out["COMPLETE_FOLLOWUP"] = out["OBS_DAYS"] >= window
    return out


def build_all(claims, report):
    episodes = build_index_episodes(claims)
    followup, sameday, excluded, index_stay = attach_followup(claims, episodes)
    summary = episode_summary(episodes, followup, sameday)
    major_events = build_major_events(followup, episodes=episodes)

    extra = [report]
    if len(excluded):
        extra.append(pd.DataFrame({
            "issue_type": "excluded_from_episode",
            "rec": excluded["rec"].values,
            "id": excluded["id"].values,
            "detail": "outside stay and not post-discharge, or PROC_DT outside 2026",
        }))
    ep_flags = episodes.melt(
        id_vars=["EPISODE_ID", "id"],
        value_vars=["INDEX_ALL_DENIED", "INDEX_MULTI_TIN", "IDENTITY_CONFLICT"],
    )
    ep_flags = ep_flags[ep_flags["value"]]
    if len(ep_flags):
        extra.append(pd.DataFrame({
            "issue_type": "episode_" + ep_flags["variable"].str.lower(),
            "rec": pd.NA,
            "id": ep_flags["id"].values,
            "detail": "episode " + ep_flags["EPISODE_ID"].values,
        }))
    incomplete = episodes[~episodes[f"COMPLETE_{config.DEFAULT_WINDOW}"]]
    if len(incomplete):
        extra.append(pd.DataFrame({
            "issue_type": "incomplete_followup",
            "rec": pd.NA,
            "id": incomplete["id"].values,
            "detail": ("episode " + incomplete["EPISODE_ID"]
                       + f" observed < {config.DEFAULT_WINDOW} days after discharge"),
        }))

    return {
        "claims_clean": claims,
        "index_episodes": episodes,
        "index_stay_claims": index_stay,
        "followup_claims": followup,
        "sameday_claims": sameday,
        "major_events": major_events,
        "episode_summary": summary,
        "data_quality_report": pd.concat([e for e in extra if len(e)], ignore_index=True),
    }


def write_outputs(outputs, out_dir=config.PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        if name != "data_quality_report":
            df = df.drop(columns=["rec"], errors="ignore")
        money = [c for c in df.columns if c in config.MONEY_COLS or c.endswith("_ALLOWED")]
        df = df.assign(**{c: df[c].round(2) for c in money})
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")


def read_outputs(out_dir=config.PROCESSED_DIR):
    """Read processed CSVs back with identifiers as text and dates parsed."""
    text_cols = {c: "string" for c in ["id", "PERSON_KEY", "BILL_TIN", "BILL_NPI", "BILL_MPIN", "REND_NPI",
                                       "INDEX_REND_NPI", "INDEX_REND_NPIS", "PROC_CD", "PLSRV_CD", "DIAG1_CD",
                                       "INDEX_DIAG1_CD", "INDEX_CPT", "EPISODE_ID", "EVENT_TYPE",
                                       "TOP_PROC_CD", "TOP_DIAG1_CD", "RELATEDNESS", "RELATEDNESS_REASON"]}
    dates = ["INDEX_SRVC_DT", "SRVC_STRT_DT", "SRVC_END_DT", "PROC_DT", "db",
             "STAY_START", "STAY_END", "START_DT", "END_DT"]
    result = {}
    names = ["index_episodes", "index_stay_claims", "followup_claims", "sameday_claims",
             "major_events", "episode_summary", "data_quality_report"]
    bool_cols = ["INPATIENT", "INDEX_ALL_DENIED", "INDEX_MULTI_TIN", "IDENTITY_CONFLICT",
                 "RELATED_TO_INDEX", "COMPLETE_30", "COMPLETE_60", "COMPLETE_90", "COMPLETE_FOLLOWUP"]
    for name in names:
        path = out_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=text_cols)
        for col in dates:
            if col in df:
                df[col] = pd.to_datetime(df[col])
        for col in bool_cols:
            if col in df:
                df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
        result[name] = df
    return result
