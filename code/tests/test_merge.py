import pandas as pd
from eegrl import merge


def test_merge_combines_shards_across_machine_dirs(tmp_path):
    a, b, out = tmp_path / "m1", tmp_path / "m2", tmp_path / "out"
    a.mkdir(); b.mkdir()
    pd.DataFrame({"report_id": ["r1", "r2"], "p_abnormal": [.1, .2]}).to_parquet(a / "x__full.parquet")
    pd.DataFrame({"report_id": ["r2", "r3"], "p_abnormal": [.2, .3]}).to_parquet(b / "x__full.parquet")
    pd.DataFrame({"report_id": ["r9"], "p_abnormal": [.9]}).to_parquet(b / "y__full.parquet")
    merge.merge_dirs([a, b], out)
    x = pd.read_parquet(out / "x__full.parquet")
    assert sorted(x.report_id) == ["r1", "r2", "r3"] and (out / "y__full.parquet").exists()
