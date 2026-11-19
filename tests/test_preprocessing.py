import re
import zipfile

import pandas as pd
import pytest

from src import config, episodes, preprocess

ROOT = config.ROOT


@pytest.fixture(scope="module")
def pipeline():
    df, raw = preprocess.load_raw()
    report = preprocess.quality_checks(df, raw)
    claims = preprocess.clean(df)
    return df, raw, claims, episodes.build_all(claims, report)


def test_raw_copy_is_identical_to_supplied_csv():
    assert (ROOT / "data.csv").read_bytes() == config.RAW_CSV.read_bytes()


def test_ids_match_values_stored_in_xlsx():
    xml = zipfile.ZipFile(ROOT / "docs" / "data.xlsx").read("xl/worksheets/sheet1.xml").decode()
    xlsx_ids = [v for _, v in re.findall(r'<c r="F(\d+)"[^>]*><v>([^<]*)</v>', xml)][1:]
    raw = pd.read_csv(config.RAW_CSV, dtype=str)
    assert raw["id"].tolist() == xlsx_ids


def test_load_shape_and_text_ids(pipeline):
    df, raw, _, _ = pipeline
    assert df.shape == (2658, 38)
    assert df["id"].nunique() == 21
    assert df["id"].str.fullmatch(r"\d+").all()
    assert (df["id"] == raw["id"].str.strip()).all()
    for col in ["prov_id", "BILL_NPI", "REND_NPI", "PROC_CD", "DIAG1_CD"]:
        assert df[col].dropna().map(type).eq(str).all()


def test_age_groups_do_not_overlap():
    for age in range(0, 100):
        hits = [g for g in config.AGE_GROUPS if g[2] <= age <= g[3]]
        assert len(hits) == 1, age
    assert preprocess.age_group(55).startswith("D")
    assert preprocess.age_group(56).startswith("E")
    assert preprocess.age_group(65).startswith("F")
    assert preprocess.age_group(pd.NA) == config.AGE_UNKNOWN


def test_processed_outputs_drop_rec_and_secondary_diagnoses(pipeline, tmp_path):
    *_, outputs = pipeline
    episodes.write_outputs(outputs, tmp_path)
    cols = pd.read_csv(tmp_path / "claims_clean.csv", nrows=1).columns
    assert "rec" not in cols
    assert not set(config.DIAG_DROP) & set(cols)
    assert {"DIAG1_CD", "DIAG1_DESC", "BILL_TIN"} <= set(cols)
    assert "prov_id" not in cols
    assert "rec" in pd.read_csv(tmp_path / "data_quality_report.csv", nrows=1).columns


def test_episodes_are_eligible_and_unique(pipeline):
    _, _, claims, outputs = pipeline
    ep = outputs["index_episodes"]
    assert ep["INDEX_SRVC_DT"].between(config.STUDY_START, config.STUDY_END).all()
    assert (ep["INDEX_CPT"] == "63280").all()
    assert ep["EPISODE_ID"].is_unique
    assert not ep.duplicated(["PERSON_KEY", "INDEX_SRVC_DT"]).any()
    idx = claims[(claims["JOURNEY_PHASE"] == "INDEX") & (claims["PROC_CD"] == "63280")
                 & claims["INDEX_SRVC_DT"].between(config.STUDY_START, config.STUDY_END)]
    assert len(ep) == idx[["PERSON_KEY", "INDEX_SRVC_DT"]].drop_duplicates().shape[0] == 16


def test_followup_strictly_after_index_and_sameday_separate(pipeline):
    *_, outputs = pipeline
    assert (outputs["followup_claims"]["DAYS_AFTER_SURGERY"] > 0).all()
    assert (outputs["sameday_claims"]["DAYS_AFTER_SURGERY"] == 0).all()
    assert (outputs["followup_claims"]["PROC_DT"].dt.year == 2026).all()


def test_identity_conflicts_are_split_not_merged(pipeline):
    _, _, claims, outputs = pipeline
    assert set(claims.loc[claims["IDENTITY_CONFLICT"], "id"]) == {"116218000000", "116219000000", "116225000000"}
    ep = outputs["index_episodes"]
    assert ep.loc[ep["id"] == "116219000000", "PERSON_KEY"].nunique() == 2


def test_quality_report_counts(pipeline):
    *_, outputs = pipeline
    counts = outputs["data_quality_report"]["issue_type"].value_counts()
    assert counts["negative_allowed"] == 21
    assert counts["zero_allowed"] == 959
    assert counts["sameday_followup"] == 539
