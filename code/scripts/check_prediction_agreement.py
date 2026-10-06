"""Exit 1 unless two prediction files agree on their common reports (same model, revision and backend)."""
import argparse
import sys
import pandas as pd
from eegrl import analyze as A

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--max-diff", type=float, default=0.02); ap.add_argument("--min-r", type=float, default=0.999)
ap.add_argument("--min-n", type=int, default=150)
x = ap.parse_args()
d = A.prediction_agreement(pd.read_parquet(x.a), pd.read_parquet(x.b))
print(d)
sys.exit(0 if d["n"] >= x.min_n and d["max_abs_diff"] <= x.max_diff and d["pearson_r"] >= x.min_r else 1)
