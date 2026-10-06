import argparse
import importlib
import pandas as pd
from eegrl import runner
from eegrl.config import INTER, PRED, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--labeler", required=True)          # module in eegrl.labelers
ap.add_argument("--name", required=True)             # output name, e.g. kev-4b
ap.add_argument("--variant", default="full", choices=["full", "masked", "impression_first"])
ap.add_argument("--splits", default="test")          # comma list; "all" = train,val,test
ap.add_argument("--cohort", default=str(INTER / "cohort.parquet"))
ap.add_argument("--pred-dir", default=str(PRED))
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--strat-sample", type=int, default=0)  # fixed hospital-stratified subset of the test split
ap.add_argument("--shard", default="")  # "i/n": keep every n-th report starting at i (0-based), after sampling
ap.add_argument("--sample", type=int, default=0)       # fixed random subset (seed SEED) of the selected splits
ap.add_argument("--opt", action="append", default=[])  # key=value passed to make_predictor
a = ap.parse_args()
df = pd.read_parquet(a.cohort, columns=["report_id", "text", "split", "hospital"])
if a.splits != "all":
    df = df[df.split.isin(a.splits.split(","))]
if a.strat_sample:
    from eegrl.speed import bench_sample
    df = bench_sample(df, a.strat_sample)
if a.sample:
    df = df.sample(a.sample, random_state=SEED).sort_index()
if a.shard:
    i, n = map(int, a.shard.split("/"))
    df = df.iloc[i::n]
opts = dict(o.split("=", 1) for o in a.opt)
fn = importlib.import_module(f"eegrl.labelers.{a.labeler}").make_predictor(**opts)
runner.run(a.name, fn, df, f"{a.pred_dir}/{a.name}__{a.variant}.parquet", a.variant, a.batch)
