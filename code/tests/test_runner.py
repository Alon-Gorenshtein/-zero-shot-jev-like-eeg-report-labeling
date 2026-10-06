import pandas as pd
from eegrl import runner


def test_resume_skips_done_and_never_duplicates(tmp_path):
    df = pd.DataFrame({"report_id": [f"r{i}" for i in range(10)], "text": ["This is a normal EEG."] * 10})
    out = tmp_path / "x.parquet"
    calls = []

    def fn(texts):
        calls.append(len(texts))
        return [{"p_abnormal": 0.1} for _ in texts]
    runner.run("dummy", fn, df.iloc[:4], out, "full", batch=3)
    runner.run("dummy", fn, df, out, "full", batch=3)
    res = pd.read_parquet(out)
    assert len(res) == 10 and res.report_id.is_unique
    assert sum(calls) == 10


def test_prepare_masked():
    df = pd.DataFrame({"report_id": ["a"], "text": ["This is an abnormal EEG. Background 6 Hz."]})
    assert "abnormal EEG" not in runner.prepare_texts(df, "masked")[0]


def test_multi_output_columns(tmp_path):
    df = pd.DataFrame({"report_id": ["a", "b"], "text": ["x.", "y."]})
    out = tmp_path / "f.parquet"
    runner.run("m", lambda t: [{"p_spikes": .9, "p_seizure": .1, "truncated": False} for _ in t], df, out, "full", batch=1)
    r = pd.read_parquet(out)
    assert {"p_spikes", "p_seizure"} <= set(r.columns) and "p_abnormal" not in r.columns and len(r) == 2
