import pandas as pd
import pytest

from src import config, episodes, metrics, preprocess


@pytest.fixture(scope="module")
def built():
    df, raw = preprocess.load_raw()
    claims = preprocess.clean(df)
    return claims, episodes.build_all(claims, preprocess.quality_checks(df, raw))


@pytest.mark.parametrize("window", config.WINDOWS)
def test_total_is_index_plus_postop(built, window):
    _, o = built
    s = episodes.episode_summary(o["index_episodes"], o["followup_claims"], o["sameday_claims"], window)
    assert len(s) == len(o["index_episodes"])
    assert (s["TOTAL_ALLOWED"] - (s["INDEX_ALLOWED"] + s["POSTOP_ALLOWED"])).abs().max() < 1e-9


def test_postop_is_monotonic_in_window_excluding_adjustments(built):
    _, o = built
    lines = [episodes.episode_summary(o["index_episodes"], o["followup_claims"], o["sameday_claims"], w)["POSTOP_LINES"]
             for w in config.WINDOWS]
    assert (lines[0] <= lines[1]).all() and (lines[1] <= lines[2]).all()


def test_hand_check_stay_based_against_pipeline(built):
    _, o = built
    ep = o["index_episodes"].set_index("EPISODE_ID")
    stay = o["index_stay_claims"]
    fu = o["followup_claims"]
    s = episodes.episode_summary(o["index_episodes"], o["followup_claims"], o["sameday_claims"], 90).set_index("EPISODE_ID")

    for ep_id in ["E003", "E005", "E007", "E012"]:
        stay_amt = stay.loc[stay["EPISODE_ID"] == ep_id, config.ALLOWED].sum()
        post = fu[(fu["EPISODE_ID"] == ep_id) & (fu["DAYS_AFTER_DISCHARGE"].between(1, 90))]
        assert ep.loc[ep_id, "INDEX_ALLOWED"] == pytest.approx(stay_amt)
        assert s.loc[ep_id, "INDEX_ALLOWED"] == pytest.approx(stay_amt)
        assert s.loc[ep_id, "POSTOP_ALLOWED"] == pytest.approx(post[config.ALLOWED].sum())
        assert s.loc[ep_id, "POSTOP_LINES"] == len(post)
        assert s.loc[ep_id, "TOTAL_ALLOWED"] == pytest.approx(stay_amt + post[config.ALLOWED].sum())


def test_cost_stats_exclude_denied_and_zero(built):
    _, o = built
    s = episodes.episode_summary(o["index_episodes"], o["followup_claims"], o["sameday_claims"], 90)
    index_stat = metrics.cost_stat_series(s, "INDEX_ALLOWED")
    postop_stat = metrics.cost_stat_series(s, "POSTOP_ALLOWED")

    assert index_stat.isna().sum() == int((s["INDEX_PAID_LINES"] == 0).sum())
    assert (index_stat.dropna() > 0).all()
    assert (postop_stat.isna() == (s["POSTOP_PAID_LINES"] == 0)).all()

    stats = metrics.burden_stats(s, "INDEX_ALLOWED")
    assert stats["n"] == int((s["INDEX_PAID_LINES"] > 0).sum())
    assert stats["median"] == pytest.approx(index_stat.median())
    assert stats["median"] > 192


def test_cohorts_cover_each_eligible_episode_once(built):
    _, o = built
    s = episodes.episode_summary(o["index_episodes"], o["followup_claims"], o["sameday_claims"], 90)
    c, lo, hi = metrics.assign_cohorts(s, 25, 75)
    eligible = metrics.cost_stat_series(s, "POSTOP_ALLOWED").notna()
    assert c.loc[eligible, "COHORT"].notna().all()
    assert c.loc[~eligible, "COHORT"].isna().all()
    assert len(c) == len(s) and c["EPISODE_ID"].is_unique
    assert metrics.cohort_summary(c)["EPISODES"].sum() == int(eligible.sum())
    assert (c.loc[c["COHORT"] == "Elevated", "POSTOP_ALLOWED"] > hi).all()


def test_threshold_changes_flags_not_data(built):
    _, o = built
    s = episodes.episode_summary(o["index_episodes"], o["followup_claims"], o["sameday_claims"], 90)
    before = s.copy()
    low, high = metrics.flag_threshold(s, 1_000), metrics.flag_threshold(s, 100_000)
    n_eligible = int(metrics.cost_stat_series(s, "POSTOP_ALLOWED").notna().sum())
    assert low["flagged"] > high["flagged"]
    assert low["total"] == high["total"] == n_eligible
    pd.testing.assert_frame_equal(s, before)


def test_proc_code_summary_reconciles(built):
    _, o = built
    fu = o["followup_claims"]
    p = metrics.proc_code_summary(fu, 90)
    in90 = fu[fu["DAYS_AFTER_DISCHARGE"] <= 90]
    eligible = metrics.cost_eligible_lines(in90)
    assert p["CLAIM_LINES"].sum() == len(in90)
    assert p["TOTAL_ALLOWED"].sum() == pytest.approx(eligible[config.ALLOWED].sum())
    assert p["DENIED_OR_ZERO_LINES"].sum() == (
        (in90["PAID_DENIED"] == "DENIED") | (in90[config.ALLOWED] == 0)
    ).sum()
    bucket_cols = [c for c in p.columns if c.startswith("ALLOWED_DAYS_")]
    assert p[bucket_cols].sum(axis=1).sum() == pytest.approx(eligible[config.ALLOWED].sum())


def test_timeline_running_total_ends_at_postop(built):
    _, o = built
    s = episodes.episode_summary(o["index_episodes"], o["followup_claims"], o["sameday_claims"], 60).set_index("EPISODE_ID")
    t = metrics.timeline(o["followup_claims"], "E012", 60)
    assert t["DAYS_AFTER_DISCHARGE"].is_monotonic_increasing
    assert t["RUNNING_ALLOWED"].iloc[-1] == pytest.approx(s.loc["E012", "POSTOP_ALLOWED"])


def test_major_events_flag_readmissions(built):
    _, o = built
    events = o["major_events"]
    assert {"subsequent_inpatient", "outpatient_surgery"} <= set(events["EVENT_TYPE"])
    e009 = events[events["EPISODE_ID"] == "E009"]
    assert (e009["EVENT_TYPE"] == "subsequent_inpatient").any()


def test_major_events_relatedness_labels(built):
    _, o = built
    events = o["major_events"]
    assert {"RELATED_TO_INDEX", "RELATEDNESS", "RELATEDNESS_REASON"} <= set(events.columns)
    assert set(events["RELATEDNESS"]) <= {"Related", "Unrelated"}
    assert events["RELATED_TO_INDEX"].dtype == bool or events["RELATED_TO_INDEX"].isin([True, False]).all()

    # Exact index DIAG1 / spine family: E009 subsequent inpatient uses D3611 (index DIAG1).
    e009_inp = events[(events["EPISODE_ID"] == "E009") & (events["EVENT_TYPE"] == "subsequent_inpatient")]
    assert len(e009_inp) >= 1
    assert bool(e009_inp.iloc[0]["RELATED_TO_INDEX"]) is True
    assert e009_inp.iloc[0]["RELATEDNESS"] == "Related"

    # Clearly outside spine/neuro family (breast / gyn / eye codes appear in pilot events).
    unrelated = events[events["TOP_DIAG1_CD"].astype(str).str.startswith(("N63", "N64", "Z01", "H34"))]
    assert len(unrelated) >= 1
    assert (unrelated["RELATEDNESS"] == "Unrelated").all()
    assert (unrelated["RELATEDNESS_REASON"] == "outside_spine_neuro_family").all()

    summary = metrics.major_event_summary(events)
    assert {"EVENT_TYPE", "RELATEDNESS", "EVENTS"} <= set(summary.columns)
    assert summary["EVENTS"].sum() == len(events)
