import glob
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from . import metrics as M, dawid_skene as DS
from .config import INTER, PRED, FINDING_COLS


def load_predictions(split="test", pred_dir=None):
    c = pd.read_parquet(INTER / "cohort.parquet").drop(columns=["text"])
    W = c[c.split == split].copy() if split != "all" else c.copy()
    for f in sorted(glob.glob(str((pred_dir or PRED) / "*__*.parquet"))):
        name = f.rsplit("/", 1)[-1][:-8]
        if name.startswith("._") or "findings" in name:
            continue
        d = pd.read_parquet(f, columns=["report_id", "p_abnormal"]).rename(columns={"p_abnormal": f"p__{name}"})
        W = W.merge(d, on="report_id", how="left")
    return W


def _cols(W):
    return [c[3:] for c in W.columns if c.startswith("p__")]


def _kappa(y, p):
    return M.threshold_metrics(y, p)["kappa"]


def primary_table(W, B=2000):
    rows = []
    for lab in _cols(W):
        y, p, cl = W.label.values, W[f"p__{lab}"].values, W.patient_id.values
        a, lo, hi = M.boot_ci(y, p, cl, M.auroc, B=B)
        k, klo, khi = M.boot_ci(y, p, cl, _kappa, B=B)
        rows.append({"labeler": lab, "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
                     "kappa": k, "kappa_lo": klo, "kappa_hi": khi, **M.threshold_metrics(y, p)})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def comparisons(W, ref, B=2000):
    rows = []
    for lab in _cols(W):
        if lab == ref:
            continue
        common = W[W[f"p__{ref}"].notna() & W[f"p__{lab}"].notna()]  # paired on reports both models scored
        d, lo, hi, pb = M.paired_diff(common.label.values, common[f"p__{ref}"].values, common[f"p__{lab}"].values,
                                      common.patient_id.values, M.auroc, B=B)
        rows.append({"ref": ref, "labeler": lab, "n_common": len(common), "diff": d, "lo": lo, "hi": hi, "p_boot": pb})
    t = pd.DataFrame(rows)
    t["q_bh"] = M.bh(t.p_boot.values)
    return t


def _strata(W):
    W = W.copy()
    W["age_group"] = np.where(W.age < 18, "pediatric", np.where(W.age >= 18, "adult", "unknown"))
    W["era"] = (W.year // 5 * 5).astype("Int64").astype(str) + "s"
    return W


def subgroups(W, labelers, B=2000):
    W = _strata(W); rows = []
    for st in ["hospital", "age_group", "age_band", "setting", "era", "service"]:
        if st not in W or W[st].isna().all():
            continue
        for lvl, g in W.groupby(st):
            if len(g) < 30:
                continue
            for lab in labelers:
                a, lo, hi = M.boot_ci(g.label.values, g[f"p__{lab}"].values, g.patient_id.values, M.auroc, B=B)
                rows.append({"stratum_type": st, "stratum": lvl, "labeler": lab, "n": len(g), "auroc": a, "lo": lo,
                             "hi": hi, "kappa": _kappa(g.label.values, g[f"p__{lab}"].values)})
    return pd.DataFrame(rows)


def _binarize(p):
    return np.where(np.isnan(p), np.nan, (p >= .5).astype(float))


def latent_class(W, labelers, B=200, seed=20260929):
    L = np.column_stack([W.label.values.astype(float)] + [_binarize(W[f"p__{l}"].values) for l in labelers])
    names = ["reference"] + list(labelers)
    r = DS.fit(L)
    rng = np.random.default_rng(seed); k, order, bounds = M._cluster_index(W.patient_id.values)
    bs = [DS.fit(L[M._resample(rng, k, order, bounds)]) for _ in range(B)]

    def ci(key, j):
        v = [b[key][j] for b in bs]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    out = {n: {"sens": float(r["sens"][j]), "sens_ci": ci("sens", j), "spec": float(r["spec"][j]), "spec_ci": ci("spec", j)}
           for j, n in enumerate(names)}
    out["prevalence"] = r["prevalence"]
    return out


def calibration(W, labelers, bins=10):
    rows = []
    for lab in labelers:
        p = W[f"p__{lab}"].values; ok = ~np.isnan(p); y = W.label.values[ok]; p = p[ok]
        idx = np.minimum((p * bins).astype(int), bins - 1)
        ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
        rows.append({"labeler": lab, "brier": float(((p - y) ** 2).mean()), "ece": float(ece)})
    return pd.DataFrame(rows)


def discordance_model(W, lab):
    W = _strata(W); W = W[~W[f"p__{lab}"].isna()]
    y = ((W[f"p__{lab}"] >= .5).astype(int) != W.label).astype(int)
    X = pd.get_dummies(W[["hospital", "age_group"]], drop_first=True).astype(float)
    X["year_z"] = (W.year - W.year.mean()) / W.year.std()
    ln = np.log(W.n_chars)
    X["len_z"] = (ln - ln.mean()) / ln.std()
    fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": W.patient_id.astype("category").cat.codes})
    ci = fit.conf_int()
    return pd.DataFrame({"term": fit.params.index, "OR": np.exp(fit.params), "lo": np.exp(ci[0]),
                         "hi": np.exp(ci[1]), "p": fit.pvalues}).reset_index(drop=True)


def disagreement_taxonomy(W, lab):
    yh = (W[f"p__{lab}"] >= .5).astype(int)
    rows = []
    for kind, mask in [("model_abnormal_ref_normal", (yh == 1) & (W.label == 0)),
                       ("model_normal_ref_abnormal", (yh == 0) & (W.label == 1))]:
        g = W[mask]
        for c in FINDING_COLS:
            rows.append({"kind": kind, "n": int(mask.sum()), "finding": c,
                         "share": float(g[f"f_{c}"].mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


_TRAIN = re.compile(r"-train-(.+?)(?:__|$)")


def train_hospital(labeler):
    m = _TRAIN.search(labeler)
    return m.group(1) if m else None


def _role(labeler, hospital):
    th = train_hospital(labeler)
    if th is None:
        return "zero-shot" if labeler.split("__")[0].startswith(("kev", "laya", "gemma", "qwen", "nli", "rule")) else "pooled"
    return "in-hospital" if hospital in th.split("+") else "out-of-hospital"


def transfer_table(W, labelers, B=2000):
    rows = []
    for lab in labelers:
        for h, g in W.groupby("hospital"):
            a, lo, hi = M.boot_ci(g.label.values, g[f"p__{lab}"].values, g.patient_id.values, M.auroc, B=B)
            rows.append({"labeler": lab, "train_hospital": train_hospital(lab) or "", "hospital": h,
                         "role": _role(lab, h), "n": len(g), "auroc": a, "lo": lo, "hi": hi,
                         "kappa": _kappa(g.label.values, g[f"p__{lab}"].values)})
    return pd.DataFrame(rows)


def _gap_stat(aucs, th):
    if th is None:
        v = [a for a in aucs.values() if np.isfinite(a)]
        return max(v) - min(v)
    ins = [a for h, a in aucs.items() if h in th.split("+")]
    outs = [a for h, a in aucs.items() if h not in th.split("+")]
    return float(np.mean(ins) - np.mean(outs))


def generalization_gap(W, lab, B=2000, seed=20260929, stat="auroc", kind="native"):
    """Supervised (trained on one hospital): in-hospital minus mean out-of-hospital AUROC.
    Zero-shot / pooled: max minus min hospital AUROC. kind="range" applies max minus min to every labeler (common metric).
    CI: patient-cluster bootstrap within each hospital."""
    fn = M.auroc if stat == "auroc" else _kappa
    th = None if kind == "range" else train_hospital(lab)
    groups = {h: g for h, g in W.groupby("hospital")}
    col = f"p__{lab}"
    est = _gap_stat({h: fn(g.label.values, g[col].values) for h, g in groups.items()}, th)
    rng = np.random.default_rng(seed)
    idx = {h: M._cluster_index(g.patient_id.values) for h, g in groups.items()}
    bs = []
    for _ in range(B):
        aucs = {}
        for h, g in groups.items():
            i = M._resample(rng, *idx[h])
            aucs[h] = fn(g.label.values[i], g[col].values[i])
        bs.append(_gap_stat(aucs, th))
    bs = np.array(bs)
    return {"labeler": lab, "stat": stat, "kind": "max_minus_min" if th is None else "in_minus_out", "gap": est,
            "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}


# ---------------- abnormality type, benign variants, findings ----------------
_CAT_ORDER = [("Seizure or status", ["seizure", "status"]),
              ("Periodic or rhythmic pattern", ["lpd", "gpd", "lrda", "grda", "bipd"]),
              ("Burst suppression", ["bs"]),
              ("Epileptiform discharges", ["spikes"]),
              ("Focal slowing only", ["foc slowing"]),
              ("Generalized slowing only", ["gen slowing"])]


def abnormality_category(W):
    """Most severe report-sourced finding of each abnormal report (hierarchical); 'Normal' for normal reports."""
    out = pd.Series("Other abnormality", index=W.index, dtype=object)
    done = pd.Series(False, index=W.index)
    for name, fs in _CAT_ORDER:
        hit = W[[f"fr_{f}" for f in fs if f"fr_{f}" in W]].sum(axis=1) > 0
        out[hit & ~done] = name
        done |= hit
    out[W.label == 0] = "Normal"
    return out


def _correct(y, p):
    ok = ~np.isnan(p)
    return float(((p[ok] >= .5).astype(int) == y[ok]).mean()) if ok.any() else np.nan


def detection_by_category(W, labelers, by="category", B=2000):
    """Per category: share of reports the model labels in agreement with the reference (sensitivity for
    abnormal categories, specificity for 'Normal')."""
    rows = []
    for lvl, g in W.groupby(by):
        for lab in labelers:
            y, p = g.label.values, g[f"p__{lab}"].values.astype(float)
            est, lo, hi = M.boot_ci(y, p, g.patient_id.values, _correct, B=B)
            rows.append({by: lvl, "labeler": lab, "n": len(g), "rate": est, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def load_findings_predictions(split="test"):
    from .config import TARGET_FINDINGS
    c = pd.read_parquet(INTER / "cohort.parquet").drop(columns=["text"])
    W = c[c.split == split].copy() if split != "all" else c.copy()
    names = []
    for f in sorted(glob.glob(str(PRED / "*findings*__full.parquet"))):
        name = f.rsplit("/", 1)[-1][: -len("__full.parquet")]
        d = pd.read_parquet(f)
        keep = {f"p_{k.replace(' ', '_')}": f"p__{name}__{k.replace(' ', '_')}" for k in TARGET_FINDINGS}
        d = d[["report_id"] + [k for k in keep if k in d]].rename(columns=keep)
        W = W.merge(d, on="report_id", how="left"); names.append(name)
    return W, names


def _findings_cell(g, col, ref_col, B):
    yy = g[ref_col].values
    a, lo, hi = M.boot_ci(yy, g[col].values, g.patient_id.values, M.auroc, B=B)
    return len(g), int(yy.sum()), a, lo, hi


def findings_table(W, labelers, findings, B=2000, n_jobs=-1):
    """AUROC (95% CI) per labeler x finding x reference (report / recording annotation) x hospital.
    Cells are independent; computed in parallel with fixed seeds, so results do not depend on n_jobs."""
    from joblib import Parallel, delayed
    tasks, keys = [], []
    for lab in labelers:
        for f in findings:
            col = f"p__{lab}__{f.replace(' ', '_')}"
            if col not in W:
                continue
            for ref in ("report", "annotation"):
                for h, g in [("All", W)] + list(W.groupby("hospital")):
                    g = g[g[col].notna()]
                    tasks.append(delayed(_findings_cell)(g[[col, f"f{ref[0]}_{f}", "patient_id"]], col, f"f{ref[0]}_{f}", B))
                    keys.append((lab, f, ref, h))
    res = Parallel(n_jobs=n_jobs)(tasks)
    return pd.DataFrame([{"labeler": lab, "finding": f, "reference": ref, "hospital": h, "n": n, "positives": p,
                          "auroc": a, "lo": lo, "hi": hi} for (lab, f, ref, h), (n, p, a, lo, hi) in zip(keys, res)])


def _verified_cell(g, col, B):
    y = np.ones(len(g), dtype=int)
    return M.boot_ci(y, g[col].values.astype(float), g.patient_id.values, _correct, B=B)


def verified_sensitivity(W, labelers, findings, B=2000, n_jobs=-1):
    """Detection rate (p >= 0.5) among HEEDB 'verified' positives (the database's highest-reliability tier,
    which has no verified negatives, so only sensitivity is estimable)."""
    from joblib import Parallel, delayed
    tasks, keys = [], []
    for lab in labelers:
        for f in findings:
            col, ref = f"p__{lab}__{f.replace(' ', '_')}", f"fv_{f}"
            if col not in W or ref not in W:
                continue
            for h, g in [("All", W)] + list(W.groupby("hospital")):
                g = g[g[ref] == 1]
                if len(g) == 0:
                    continue
                tasks.append(delayed(_verified_cell)(g[[col, "patient_id"]], col, B)); keys.append((lab, f, h, len(g)))
    res = Parallel(n_jobs=n_jobs)(tasks)
    return pd.DataFrame([{"labeler": lab, "finding": f, "hospital": h, "n_verified": n, "sensitivity": est, "lo": lo, "hi": hi}
                         for (lab, f, h, n), (est, lo, hi) in zip(keys, res)])


def _latent_class_one(f, cols, pid, B, seed):
    L = np.column_stack([c for _, c in cols])
    r = DS.fit(L)
    rng = np.random.default_rng(seed); k, order, bounds = M._cluster_index(pid)
    bs = [DS.fit(L[M._resample(rng, k, order, bounds)]) for _ in range(B)]
    rows = []
    for j, (name, _) in enumerate(cols):
        sv, sp = [b["sens"][j] for b in bs], [b["spec"][j] for b in bs]
        rows.append({"finding": f, "source": name, "sens": float(r["sens"][j]), "sens_lo": float(np.percentile(sv, 2.5)),
                     "sens_hi": float(np.percentile(sv, 97.5)), "spec": float(r["spec"][j]),
                     "spec_lo": float(np.percentile(sp, 2.5)), "spec_hi": float(np.percentile(sp, 97.5)),
                     "prevalence": r["prevalence"]})
    return rows


def latent_class_findings(W, labelers, findings, B=200, seed=20260929, n_jobs=-1):
    """Per finding: Dawid-Skene over the report reference, recording annotations and every model (binarized at 0.5).
    Returns estimated sensitivity/specificity of each source, including the report reference itself.
    Findings are fitted in parallel; each uses the same seed, so results do not depend on n_jobs."""
    from joblib import Parallel, delayed
    tasks = []
    for f in findings:
        key = f.replace(" ", "_")
        cols = [("report reference", W[f"fr_{f}"].values.astype(float)),
                ("recording annotations", W[f"fa_{f}"].values.astype(float))]
        cols += [(lab, _binarize(W[f"p__{lab}__{key}"].values.astype(float))) for lab in labelers if f"p__{lab}__{key}" in W]
        tasks.append(delayed(_latent_class_one)(f, cols, W.patient_id.values, B, seed))
    return pd.DataFrame([r for rows in Parallel(n_jobs=n_jobs)(tasks) for r in rows])


SUBSET_ONLY = ("kev-27b",)  # runs defined only on the fixed 6,000-report subset (extra rows from an aborted full run)


def restrict_to_subset(W, subset_ids, prefixes=SUBSET_ONLY):
    W = W.copy()
    outside = ~W.report_id.isin(subset_ids)
    for c in W.columns:
        if c.startswith("p__") and c[3:].startswith(prefixes):
            W.loc[outside, c] = np.nan
    return W


def consensus_audit(W, labelers, z=1.959964):
    """Per hospital and reference class: share of reports where every listed zero-shot labeler (p >= 0.5)
    contradicts the reference label unanimously, with a Wilson 95% CI. Descriptive; not a gold standard."""
    P = np.column_stack([W[f"p__{l}"].values.astype(float) for l in labelers])
    ok = ~np.isnan(P).any(axis=1)
    call_abn, call_nrm = (P >= 0.5).all(axis=1), (P < 0.5).all(axis=1)
    rows = []
    for h, idx in W[ok].groupby("hospital").groups.items():
        i = W.index.get_indexer(idx)
        for ref, val, against in (("normal", 0, call_abn), ("abnormal", 1, call_nrm)):
            m = i[W.label.values[i] == val]
            n, k = len(m), int(against[m].sum())
            p = k / n if n else np.nan
            c = (p + z * z / (2 * n)) / (1 + z * z / n) if n else np.nan
            hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else np.nan
            rows.append({"hospital": h, "reference": ref, "n": n, "k": k, "prop": p, "lo": c - hw, "hi": c + hw})
    return pd.DataFrame(rows)



def external_paired(W, zero_shot, supervised, B=2000, seed=20260929):
    """For each single-hospital supervised labeler and each hospital it was NOT trained at: AUROC(zero-shot) minus
    AUROC(supervised) on the same external reports (rows scored by both), patient-clustered paired bootstrap.
    Benjamini-Hochberg across the whole family."""
    rows = []
    for sup in supervised:
        th = train_hospital(sup)
        for h, g in W.groupby("hospital"):
            if th is None or h in th.split("+"):
                continue
            for z in zero_shot:
                d = g[g[f"p__{z}"].notna() & g[f"p__{sup}"].notna()]
                diff, lo, hi, p = M.paired_diff(d.label.values, d[f"p__{z}"].values, d[f"p__{sup}"].values,
                                                d.patient_id.values, M.auroc, B=B, seed=seed)
                rows.append({"zero_shot": z, "supervised": sup, "train_hospital": th, "target_hospital": h, "n": len(d),
                             "auroc_zero_shot": M.auroc(d.label.values, d[f"p__{z}"].values),
                             "auroc_supervised": M.auroc(d.label.values, d[f"p__{sup}"].values),
                             "diff": diff, "lo": lo, "hi": hi, "p_boot": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = M.bh(out.p_boot.values)
    return out


def youden_threshold(y, p):
    """Threshold maximizing sensitivity + specificity - 1 (ties: the lowest such threshold)."""
    y = np.asarray(y); p = np.asarray(p, float); ok = ~np.isnan(p); y, p = y[ok], p[ok]
    cand = np.unique(p)
    pos, neg = y.sum(), len(y) - y.sum()
    order = np.argsort(p); ys = y[order]; ps = p[order]
    # predicted positive when p >= t; for each unique t, count positives/negatives at or above it
    idx = np.searchsorted(ps, cand, side="left")
    tp = pos - np.concatenate([[0], np.cumsum(ys)])[idx]
    fp = neg - np.concatenate([[0], np.cumsum(1 - ys)])[idx]
    j = tp / pos + (1 - fp / neg) - 1
    return float(cand[int(np.argmax(j))])


def transported_threshold(W, lab, t):
    """Apply one frozen threshold to every hospital: sensitivity, specificity, Cohen kappa, balanced accuracy."""
    rows = []
    for h, g in W[W[f"p__{lab}"].notna()].groupby("hospital"):
        y = g.label.values; yhat = (g[f"p__{lab}"].values >= t).astype(int)
        tp = ((yhat == 1) & (y == 1)).sum(); tn = ((yhat == 0) & (y == 0)).sum()
        sens = tp / max(y.sum(), 1); spec = tn / max((1 - y).sum(), 1)
        po = (yhat == y).mean(); pe = yhat.mean() * y.mean() + (1 - yhat.mean()) * (1 - y.mean())
        rows.append({"labeler": lab, "hospital": h, "threshold": t, "n": len(g), "sens": sens, "spec": spec,
                     "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan, "bal_acc": (sens + spec) / 2})
    return pd.DataFrame(rows)


def hospital_calibration(W, labelers, bins=10):
    """Per labeler and hospital: Brier, ECE, calibration intercept (calibration-in-the-large, slope fixed at 1)
    and calibration slope (logistic regression of the label on logit(p))."""
    import statsmodels.api as sm
    rows = []
    for lab in labelers:
        for h, g in W[W[f"p__{lab}"].notna()].groupby("hospital"):
            y = g.label.values.astype(float); p = np.clip(g[f"p__{lab}"].values.astype(float), 1e-4, 1 - 1e-4)
            lp = np.log(p / (1 - p))
            idx = np.minimum((p * bins).astype(int), bins - 1)
            ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
            try:
                f1 = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
                f0 = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
                slope = f1.params[1] if f1.converged else np.nan
                icpt = f0.params[0] if f0.converged and abs(f0.params[0]) < 50 else np.nan
            except Exception:
                slope = icpt = np.nan
            rows.append({"labeler": lab, "hospital": h, "n": len(g), "brier": float(((p - y) ** 2).mean()),
                         "ece": float(ece), "intercept": float(icpt), "slope": float(slope)})
    return pd.DataFrame(rows)


def wilson(k, n, z=1.959964):
    """Wilson score 95% interval for k of n; (nan, nan, nan) when n == 0."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - hw, c + hw


def mention_annotation_check(C, finding, classify):
    """Post hoc, non-text corroboration of the rule-based mention audit: by reference label and by how the report
    mentions the finding, the share of reports with that finding annotated in the recording (Wilson 95% CI)."""
    C = C.assign(_m=C.text.map(lambda t: classify(t, finding)))
    rows = []
    for h, g in [("All", C)] + list(C.groupby("hospital")):
        for (ref, m), gg in g.groupby([f"fr_{finding}", "_m"]):
            k, n = int(gg[f"fa_{finding}"].sum()), len(gg)
            p, lo, hi = wilson(k, n)
            rows.append({"finding": finding, "hospital": h, "reference": int(ref), "mention": m,
                         "n": n, "annotated": k, "prop": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def consensus_annotation_check(W, labelers, annotated):
    """Post hoc: among reference-normal reports per hospital, those every listed zero-shot labeler calls abnormal
    (p >= 0.5) vs the rest; share with epileptiform, periodic or seizure activity annotated in the recording."""
    P = np.column_stack([W[f"p__{l}"].values.astype(float) for l in labelers])
    ok = ~np.isnan(P).any(axis=1) & (W.label.values == 0)
    D = W[ok].assign(_c=(P[ok] >= 0.5).all(axis=1))
    rows = []
    for h, g in D.groupby("hospital"):
        for grp, gg in (("concordant abnormal call", g[g._c]), ("other reference-normal", g[~g._c])):
            k, n = int(gg[annotated].sum()), len(gg)
            p, lo, hi = wilson(k, n)
            rows.append({"hospital": h, "group": grp, "n": n, "annotated": k, "prop": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def source_thresholds(V, lab, hospitals):
    """Youden threshold of one labeler from each source hospital's validation reports: {hospital: threshold}."""
    out = {}
    for h in hospitals:
        v = V[V.hospital == h]
        if v.empty or v[f"p__{lab}"].isna().all():
            raise ValueError(f"no validation scores for {lab} at {h}")
        if v[f"p__{lab}"].isna().any():  # a partial run would quietly bias the threshold
            raise ValueError(f"{int(v[f'p__{lab}'].isna().sum())} validation scores missing for {lab} at {h}")
        out[h] = youden_threshold(v.label.values, v[f"p__{lab}"].values)
    return out


def prediction_agreement(a, b):
    """Agreement of two prediction files on their common reports: n, max |difference|, Pearson r."""
    m = a[["report_id", "p_abnormal"]].merge(b[["report_id", "p_abnormal"]], on="report_id", suffixes=("_a", "_b")).dropna()
    d = (m.p_abnormal_a - m.p_abnormal_b).abs()
    r = float(np.corrcoef(m.p_abnormal_a, m.p_abnormal_b)[0, 1]) if len(m) > 1 else np.nan
    return {"n": len(m), "max_abs_diff": float(d.max()) if len(m) else np.nan, "pearson_r": r}


def mention_model_check(C, finding, classify, preds):
    """Post hoc, model-based check of the rule-based mention audit: by reference label and mention class, the share of
    reports each zero-shot labeler called positive for the finding (p >= 0.5), with Wilson 95% CI, on the reports that
    labeler scored. preds: {labeler: DataFrame with report_id and p_<finding>}."""
    col = f"p_{finding}"
    C = C.assign(_m=C.text.map(lambda t: classify(t, finding)))
    rows = []
    for lab, P in preds.items():
        if P[col].isna().any():
            raise ValueError(f"{lab}: {int(P[col].isna().sum())} {col} scores missing")
        D = C.merge(P[["report_id", col]], on="report_id", how="inner")
        for (ref, m), g in D.groupby([f"fr_{finding}", "_m"]):
            k, n = int((g[col] >= 0.5).sum()), len(g)
            p, lo, hi = wilson(k, n)
            rows.append({"finding": finding, "labeler": lab, "reference": int(ref), "mention": m, "n": n,
                         "called_present": k, "prop": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
