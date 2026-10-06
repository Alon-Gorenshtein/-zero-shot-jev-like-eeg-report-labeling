import numpy as np
import pandas as pd
from eegrl.split import assign_split


def _df():
    rng = np.random.default_rng(0)
    rows = []
    for site in ["A", "B"]:
        for p in range(500):
            for _ in range(rng.integers(1, 4)):
                rows.append({"site": site, "patient_id": f"{site}{p}", "label": int(rng.random() < .6)})
    return pd.DataFrame(rows)


def test_no_patient_crosses_splits_and_fractions_per_site():
    df = _df()
    s = assign_split(df)
    assert (df.assign(s=s).groupby("patient_id").s.nunique() == 1).all()
    for site, g in df.assign(s=s).groupby("site"):
        pf = g.drop_duplicates("patient_id").s.value_counts(normalize=True)
        assert abs(pf["test"] - .2) < .03 and abs(pf["val"] - .1) < .03


def test_deterministic():
    df = _df()
    assert assign_split(df).equals(assign_split(df))
