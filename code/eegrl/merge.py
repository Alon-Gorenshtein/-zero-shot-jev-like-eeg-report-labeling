"""Combine prediction files pulled from several workers (shards of the same run share a file name)."""
from collections import defaultdict
from pathlib import Path
import pandas as pd


def merge_dirs(dirs, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    files = defaultdict(list)
    for d in dirs:
        for f in Path(d).glob("*.parquet"):
            if not f.name.startswith("._") and ".part" not in f.name:
                files[f.name].append(f)
    for name, fs in files.items():
        pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("report_id").to_parquet(out / name, index=False)
    return sorted(files)
