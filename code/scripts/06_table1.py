"""Table 1: cohort characteristics by hospital (all labeled reports and the test split)."""
import pandas as pd
from eegrl.config import OUT

c = pd.read_parquet(OUT / "intermediate" / "cohort.parquet")


def summarize(d):
    a = d.age_days / 365.25
    return pd.Series({"reports": len(d), "patients": d.patient_id.nunique(), "abnormal_n": int(d.label.sum()),
                      "abnormal_pct": 100 * d.label.mean(), "age_median_y": a.median(), "age_q1": a.quantile(.25),
                      "age_q3": a.quantile(.75), "female_pct": 100 * (d.sex == "F").mean(),
                      "year_first": d.year.min(), "year_last": d.year.max(),
                      "words_median": d.text.str.split().str.len().median(),
                      **{f"setting_{k}_pct": 100 * v for k, v in d.setting.value_counts(normalize=True).items()}})


rows = []
for scope, d in (("all", c), ("test", c[c.split == "test"])):
    for h, g in [("All", d)] + list(d.groupby("hospital")):
        rows.append(summarize(g).rename(f"{scope}|{h}"))
t = pd.DataFrame(rows).fillna(0).round(1)
t.to_csv(OUT / "tables" / "table1_cohort.csv")
print(t.T.to_string())
