import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from eegrl import metrics as M


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1); y = rng.integers(0, 2, 500); p = np.round(rng.random(500), 1)
    assert abs(M.auroc(y, p) - roc_auc_score(y, p)) < 1e-12


def test_auroc_single_class_nan():
    assert np.isnan(M.auroc(np.ones(5), np.random.rand(5)))


def test_threshold_metrics_and_abstain():
    y = np.array([1, 1, 0, 0, 1]); p = np.array([.9, .2, .1, .7, np.nan])
    m = M.threshold_metrics(y, p)
    assert m["abstain"] == .2 and abs(m["sens"] - .5) < 1e-12 and abs(m["spec"] - .5) < 1e-12
    assert abs(m["kappa"] - cohen_kappa_score(y[:4], (p[:4] >= .5).astype(int))) < 1e-12


def test_boot_ci_contains_estimate_and_handles_degenerate():
    rng = np.random.default_rng(2); y = rng.integers(0, 2, 400); p = y * .6 + rng.random(400) * .5
    cl = np.repeat(np.arange(200), 2)
    est, lo, hi = M.boot_ci(y, p, cl, M.auroc, B=200)
    assert lo <= est <= hi
    est, lo, hi = M.boot_ci(np.ones(10), np.random.rand(10), np.arange(10), M.auroc, B=50)
    assert np.isnan(est)


def test_paired_diff_sign():
    rng = np.random.default_rng(3); y = rng.integers(0, 2, 600)
    good = y + rng.normal(0, .3, 600); bad = y + rng.normal(0, 1.5, 600)
    d, lo, hi, pb = M.paired_diff(y, good, bad, np.arange(600), M.auroc, B=300)
    assert d > 0 and lo > 0 and pb < .05


def test_bh():
    q = M.bh(np.array([.01, .04, .03, .5]))
    assert np.allclose(q, [.04, .0533333, .0533333, .5], atol=1e-6)
