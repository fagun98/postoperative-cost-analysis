"""Validate stay-based index logic on E001-E010 and write a supervisor summary table.

Run:  python scripts/validate_sample_stays.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, episodes, preprocess  # noqa: E402

SAMPLE_IDS = [f"E{i:03d}" for i in range(1, 11)]
OUT = config.PROCESSED_DIR / "sample_10_case_summary.csv"


def main():
    df, raw = preprocess.load_raw()
    claims = preprocess.clean(df)
    built = episodes.build_all(claims, preprocess.quality_checks(df, raw))
    ep = built["index_episodes"].set_index("EPISODE_ID")
    summary = built["episode_summary"].set_index("EPISODE_ID")
    events = built["major_events"]

    stay = built["index_stay_claims"]
    mislabeled = (
        stay[stay["DAYS_AFTER_SURGERY"] > 0]
        .groupby("EPISODE_ID")[config.ALLOWED]
        .sum()
        .reindex(SAMPLE_IDS)
        .fillna(0.0)
    )

    rows = []
    for eid in SAMPLE_IDS:
        e = ep.loc[eid]
        s = summary.loc[eid]
        ev = events[events["EPISODE_ID"] == eid] if len(events) else events
        rows.append({
            "EPISODE_ID": eid,
            "SURGERY_DT": e["INDEX_SRVC_DT"].date().isoformat(),
            "STAY_START": e["STAY_START"].date().isoformat(),
            "STAY_END": e["STAY_END"].date().isoformat(),
            "INPATIENT": bool(e["INPATIENT"]),
            "LOS_DAYS": int(e["LOS_DAYS"]),
            "INDEX_ALLOWED_STAY": round(float(s["INDEX_ALLOWED"]), 2),
            "INDEX_CPT_ONLY": round(float(e["INDEX_CPT_ALLOWED"]), 2),
            "RECLASSIFIED_INTO_INDEX": round(float(mislabeled.loc[eid]), 2),
            "POSTOP_ALLOWED_90D": round(float(s["POSTOP_ALLOWED"]), 2),
            "SUBSEQUENT_INPATIENT_EVENTS": int(s["SUBSEQUENT_INPATIENT_EVENTS"]),
            "OUTPATIENT_SURGERY_EVENTS": int(s["OUTPATIENT_SURGERY_EVENTS"]),
            "MAJOR_EVENT_ALLOWED": round(float(s["MAJOR_EVENT_ALLOWED"]), 2),
            "MAJOR_EVENT_DETAIL": "; ".join(
                f"{r.EVENT_TYPE} {pd.Timestamp(r.START_DT).date()} ${r.ALLOWED:,.0f}"
                for r in ev.itertuples()
            ) if len(ev) else "",
        })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {OUT}")
    print(f"Total reclassified into index (was postop under surgery-day rule): "
          f"${mislabeled.sum():,.2f}")


if __name__ == "__main__":
    main()
