import zipfile
import pandas as pd
import pytest
from eegrl import cohort


def _csv(tmp_path, rows, name="x.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


BASE = {"BDSPPatientID": 1, "AgeAtVisit": 30, "SexDSC": "F"}


def test_dedupes_sessions_to_one_row_per_report(tmp_path):
    p = _csv(tmp_path, [
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_1_1_20180101.txt", "normal": None, "abnormal": "report", "spikes": "report"},
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_1_1_20180101.txt", "normal": None, "abnormal": "report", "spikes": None},
    ])
    df = cohort.load_label_table({("I9", ""): p})
    assert len(df) == 1
    assert df.label.iloc[0] == 1
    assert df.year.iloc[0] == 2018
    assert df["f_spikes"].iloc[0] == 1  # any-session OR


def test_conflicting_labels_raise(tmp_path):
    p = _csv(tmp_path, [
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_1_1_20180101.txt", "normal": "report", "abnormal": None},
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_1_1_20180101.txt", "normal": None, "abnormal": "report"},
    ])
    with pytest.raises(ValueError, match="conflicting"):
        cohort.load_label_table({("I9", ""): p})


def test_excludes_neither_label(tmp_path):
    p = _csv(tmp_path, [
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_2_1_20180101.txt", "normal": None, "abnormal": None},
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_3_1_20180101.txt", "normal": "report", "abnormal": None},
    ])
    df = cohort.load_label_table({("I9", ""): p})
    assert list(df.report_id) == ["NeuroReport_3_1_20180101.txt"]
    assert df.label.iloc[0] == 0


def test_zip_index_skips_macosx_and_reads_bad_bytes(tmp_path):
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("2021/NeuroReport_9_9_20210101.txt", b"This is an abnormal EEG.\r\nline\xff two")
        f.writestr("__MACOSX/2021/._NeuroReport_9_9_20210101.txt", b"\x00\x05junk")
    idx = cohort.build_zip_index([z])
    assert idx == {"NeuroReport_9_9_20210101.txt": str(z)}
    txt = cohort.read_reports(["NeuroReport_9_9_20210101.txt"], idx)["NeuroReport_9_9_20210101.txt"]
    assert txt.startswith("This is an abnormal EEG.")
    assert "\r" not in txt


def test_finding_sources_are_separated(tmp_path):
    p = _csv(tmp_path, [
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_1_1_20180101.txt", "normal": None, "abnormal": "report",
         "spikes": "annotation", "lpd": "report annotation", "posts": "report"},
        {**BASE, "DeidentifiedName(Reports)": "NeuroReport_1_1_20180101.txt", "normal": None, "abnormal": "report",
         "spikes": "annotation verified", "lpd": None, "posts": None},
    ])
    r = cohort.load_label_table({("I9", ""): p}).iloc[0]
    assert r["fr_spikes"] == 0 and r["fa_spikes"] == 1
    assert r["fr_lpd"] == 1 and r["fa_lpd"] == 1
    assert r["fr_posts"] == 1 and r["fa_posts"] == 0
    assert r["fv_spikes"] == 1 and r["fv_lpd"] == 0
