"""Controlled speed benchmark: single-report latency and batched throughput on a fixed stratified sample."""
import time
import numpy as np
import pandas as pd
from .config import SEED


def bench_sample(df, n=1000):
    t = df[df.split == "test"]
    hs = sorted(t.hospital.unique())
    k = {h: n // len(hs) + (1 if i < n % len(hs) else 0) for i, h in enumerate(hs)}
    parts = [g.sample(k[h], random_state=SEED) for h, g in t.groupby("hospital")]
    return pd.concat(parts).sort_values("report_id").reset_index(drop=True)


def summarize(latencies, batch_wall, n_batch):
    lat = np.asarray(latencies, float)
    rps = n_batch / batch_wall
    return {"latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
            "throughput_rps": float(rps), "hours_per_100k": 100000 / rps / 3600}


def run(predict_single, predict_batch, texts, n_latency=100, warmup=5):
    for t in texts[:warmup]:
        predict_single([t])
    lat = []
    for t in texts[:n_latency]:
        t0 = time.perf_counter(); predict_single([t]); lat.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); predict_batch(texts); wall = time.perf_counter() - t0
    return summarize(lat, wall, len(texts))
