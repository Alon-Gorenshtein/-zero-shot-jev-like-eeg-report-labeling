import numpy as np


def fit(L, max_iter=200, tol=1e-8):
    """Binary Dawid-Skene EM. L: items x labelers, values {0, 1, NaN}."""
    L = np.asarray(L, float); obs = ~np.isnan(L); X = np.nan_to_num(L)
    post = np.nanmean(L, axis=1)                      # majority-vote init
    post = np.where(np.isnan(post), .5, post)
    ll_old = -np.inf
    for _ in range(max_iter):
        pi = post.mean()
        w1, w0 = post[:, None] * obs, (1 - post)[:, None] * obs
        sens = np.clip((w1 * X).sum(0) / w1.sum(0), 1e-6, 1 - 1e-6)
        spec = np.clip((w0 * (1 - X)).sum(0) / w0.sum(0), 1e-6, 1 - 1e-6)
        l1 = np.log(pi) + (obs * (X * np.log(sens) + (1 - X) * np.log(1 - sens))).sum(1)
        l0 = np.log(1 - pi) + (obs * ((1 - X) * np.log(spec) + X * np.log(1 - spec))).sum(1)
        m = np.maximum(l1, l0); ll = (m + np.log(np.exp(l1 - m) + np.exp(l0 - m))).sum()
        post = 1 / (1 + np.exp(l0 - l1))
        if ll - ll_old < tol:
            break
        ll_old = ll
    return {"prevalence": float(post.mean()), "sens": sens, "spec": spec, "posterior": post, "loglik": float(ll)}
