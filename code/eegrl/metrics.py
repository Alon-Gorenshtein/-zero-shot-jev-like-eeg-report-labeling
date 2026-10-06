import numpy as np
from scipy.stats import rankdata
from .config import SEED


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p, float)
    ok = ~np.isnan(p); y, p = y[ok], p[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def threshold_metrics(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p, float)
    ok = ~np.isnan(p); yy, yh = y[ok], (p[ok] >= thr).astype(int)
    tp = int(((yy == 1) & (yh == 1)).sum()); tn = int(((yy == 0) & (yh == 0)).sum())
    fp = int(((yy == 0) & (yh == 1)).sum()); fn = int(((yy == 1) & (yh == 0)).sum())
    d = lambda a, b: a / b if b else np.nan
    n = tp + tn + fp + fn
    po = d(tp + tn, n)
    pe = d((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), n * n)
    sens, spec = d(tp, tp + fn), d(tn, tn + fp)
    return {"sens": sens, "spec": spec, "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn),
            "bal_acc": (sens + spec) / 2, "kappa": d(po - pe, 1 - pe), "abstain": float((~ok).mean()), "n": n}


def _cluster_index(clusters):
    u, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(u) + 1))
    return len(u), order, bounds


def _resample(rng, k, order, bounds):
    pick = rng.integers(0, k, k)
    return np.concatenate([order[bounds[c]:bounds[c + 1]] for c in pick])


def boot_ci(y, p, clusters, stat_fn, B=2000, seed=SEED):
    y = np.asarray(y); p = np.asarray(p, float)
    est = stat_fn(y, p)
    if np.isnan(est):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); k, order, bounds = _cluster_index(np.asarray(clusters))
    bs = np.array([stat_fn(y[i], p[i]) for i in (_resample(rng, k, order, bounds) for _ in range(B))])
    bs = bs[~np.isnan(bs)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_diff(y, p1, p2, clusters, stat_fn, B=2000, seed=SEED):
    y = np.asarray(y); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d0 = stat_fn(y, p1) - stat_fn(y, p2)
    rng = np.random.default_rng(seed); k, order, bounds = _cluster_index(np.asarray(clusters))
    ds = []
    for _ in range(B):
        i = _resample(rng, k, order, bounds)
        ds.append(stat_fn(y[i], p1[i]) - stat_fn(y[i], p2[i]))
    ds = np.array(ds); ds = ds[~np.isnan(ds)]
    p_boot = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), p_boot


def bh(pvals):
    p = np.asarray(pvals, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out
