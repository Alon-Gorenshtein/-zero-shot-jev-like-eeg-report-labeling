import numpy as np
import pandas as pd
from eegrl import speed


def test_summarize_latency_and_projection():
    s = speed.summarize(latencies=[0.1, 0.2, 0.3, 0.4], batch_wall=10.0, n_batch=1000)
    assert abs(s["latency_p50_s"] - 0.25) < 1e-9 and s["latency_p95_s"] > 0.35
    assert s["throughput_rps"] == 100.0
    assert abs(s["hours_per_100k"] - 100000 / 100 / 3600) < 1e-9


def test_benchmark_sample_is_stratified_and_fixed():
    df = pd.DataFrame({"report_id": [f"r{i}" for i in range(3000)], "split": "test",
                       "hospital": np.repeat(["A", "B", "C"], 1000)})
    a, b = speed.bench_sample(df, 300), speed.bench_sample(df, 300)
    assert list(a.report_id) == list(b.report_id) and (a.hospital.value_counts() == 100).all()


def test_benchmark_sample_exact_size_when_not_divisible():
    df = pd.DataFrame({"report_id": [f"r{i}" for i in range(3000)], "split": "test",
                       "hospital": np.repeat(["A", "B", "C"], 1000)})
    s = speed.bench_sample(df, 1000)
    assert len(s) == 1000 and s.hospital.value_counts().max() - s.hospital.value_counts().min() <= 1
