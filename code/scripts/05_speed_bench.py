"""Controlled speed benchmark on a fixed 1,000-report stratified test sample (run on an otherwise idle machine).
Example: python scripts/05_speed_bench.py --labeler systemone --name kev-4b --machine dgx-spark \
         --params 4.0e9 --opt url=http://127.0.0.1:8009 --batch-opt workers=8 [--task findings]"""
import argparse
import importlib
import socket
from pathlib import Path
import pandas as pd
from eegrl import speed
from eegrl.config import INTER, OUT
from eegrl.runner import prepare_texts

ap = argparse.ArgumentParser()
ap.add_argument("--labeler", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--machine", default=socket.gethostname()); ap.add_argument("--params", type=float, default=float("nan"))
ap.add_argument("--task", default="binary", choices=["binary", "findings"])
ap.add_argument("--cohort", default=str(INTER / "cohort.parquet")); ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--opt", action="append", default=[]); ap.add_argument("--batch-opt", action="append", default=[])
ap.add_argument("--out", default=str(OUT / "tables" / "speed_benchmark.csv"))
a = ap.parse_args()
df = speed.bench_sample(pd.read_parquet(a.cohort, columns=["report_id", "text", "split", "hospital"]), a.n)
texts = prepare_texts(df, "full")
mod = importlib.import_module(f"eegrl.labelers.{a.labeler}")
base = dict(o.split("=", 1) for o in a.opt) | ({"task": "findings"} if a.task == "findings" else {})
single = mod.make_predictor(**base)
batch = mod.make_predictor(**(base | dict(o.split("=", 1) for o in a.batch_opt))) if a.batch_opt else single
row = {"model": a.name, "task": a.task, "machine": a.machine, "params": a.params, "n": len(texts),
       **speed.run(single, batch, texts)}
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
print(row)
