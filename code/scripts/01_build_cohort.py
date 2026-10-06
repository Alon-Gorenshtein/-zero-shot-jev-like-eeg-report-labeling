from eegrl import cohort
from eegrl.config import INTER
from eegrl.split import assign_split

INTER.mkdir(parents=True, exist_ok=True)
df = cohort.load_label_table()
counts = df.groupby("site").label.agg(["size", "sum"])
print(counts)
assert len(df) == 109_339, len(df)
assert int(df.label.sum()) == 73_238
idx = cohort.build_zip_index()
missing = set(df.report_id) - set(idx)
assert not missing, f"{len(missing)} reports missing from zips"
texts = cohort.read_reports(df.report_id.tolist(), idx)
df["text"] = df.report_id.map(texts)
df["n_chars"] = df.text.str.len()
assert df.text.notna().all()
empty = df.n_chars < 50
print("empty texts by site:", df[empty].site.value_counts().to_dict())
# I0002 archive entries are 0-byte files for every labeled report (text unrecoverable); see ledger ruling.
assert set(df[empty].site) <= {"I0002"}, "unexpected empty reports outside I0002"
df = df[~empty].reset_index(drop=True)
assert len(df) == 99_791 and int(df.label.sum()) == 64_976, (len(df), int(df.label.sum()))
df["hospital"] = df.site + df.subsite.where(df.subsite == "", "-" + df.subsite)
df["split"] = assign_split(df)
from eegrl.text import setting, age_band
df["setting"] = [setting(t, s) for t, s in zip(df.text, df.service)]
df["age_band"] = [age_band(a) for a in df.age_days]
df.to_parquet(INTER / "cohort.parquet", index=False)
print(df.groupby(["site", "split"]).size().unstack())
