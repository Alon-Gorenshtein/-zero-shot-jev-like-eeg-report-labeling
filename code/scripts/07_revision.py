"""Revision analyses (post hoc, after statistical review):
1. common cross-hospital range (max - min AUROC) for every labeler;
2. paired external-hospital AUROC differences, zero-shot minus supervised-trained-elsewhere, on the same reports;
3. thresholds selected on each supervised model's source-hospital validation set (Youden), frozen and transported;
4. hospital-specific calibration (Brier, ECE, intercept, slope)."""
import json
import pandas as pd
from joblib import Parallel, delayed
from eegrl import analyze as A
from eegrl.config import OUT

T = OUT / "tables"; B = 2000
W = A.restrict_to_subset(A.load_predictions("test"), pd.read_csv(T / "subset6000_report_ids.csv").report_id)
ZS = ["kev-0.8b__full", "kev-4b__full", "kev-9b__full", "kev-27b__full", "laya-en__full", "laya-multi__full",
      "nli-deberta__full", "gemma4-e4b__full", "gemma4-12b__full", "qwen38-27b__full", "rule__full"]
SUP1 = [f"{m}-train-{h}__full" for m in ("tfidf", "clinicalbert") for h in ("I0001-S0001", "I0001-S0002", "I0003")]
POOLED = ["tfidf__full", "clinicalbert__full"]
ZS = [z for z in ZS if f"p__{z}" in W]

# 1. common range
rng = Parallel(n_jobs=-1)(delayed(A.generalization_gap)(W, l, B=B, kind="range") for l in ZS + SUP1 + POOLED)
pd.DataFrame(rng).to_csv(T / "generalization_gap_range.csv", index=False)

# 2. paired external comparisons (main family: decision models + NLI + LLMs; rule excluded because it abstains)
zs_main = [z for z in ZS if z != "rule__full"]
parts = Parallel(n_jobs=-1)(delayed(A.external_paired)(W, [z], [s], B=B) for z in zs_main for s in SUP1)
ext = pd.concat(parts, ignore_index=True)
ext["q_bh"] = A.M.bh(ext.p_boot.values)
ext.to_csv(T / "external_paired.csv", index=False)

# 3. thresholds frozen from the source-hospital validation set
V = A.load_predictions("val", pred_dir=OUT / "predictions_val")
rows = []
for lab in SUP1 + POOLED:
    th = A.train_hospital(lab)
    if f"p__{lab}" not in V:
        continue
    v = V[V[f"p__{lab}"].notna()]
    if th is not None:
        v = v[v.hospital.isin(th.split("+"))]
    t = A.youden_threshold(v.label.values, v[f"p__{lab}"].values)
    rows.append(A.transported_threshold(W, lab, t).assign(rule="source validation (Youden)"))
    rows.append(A.transported_threshold(W, lab, .5).assign(rule="fixed 0.5"))
HOSP = ["I0001-S0001", "I0001-S0002", "I0003"]
SYM = ["kev-4b__full", "kev-9b__full", "laya-en__full", "nli-deberta__full"]
if missing := [l for l in SYM if f"p__{l}" not in V]:
    raise SystemExit(f"validation scores missing for {missing}")
for lab in ZS:
    rows.append(A.transported_threshold(W, lab, .5).assign(rule="native 0.5"))
    if lab in SYM:  # symmetric sensitivity analysis (post hoc): zero-shot thresholds from each source hospital
        for src, t in A.source_thresholds(V, lab, HOSP).items():
            rows.append(A.transported_threshold(W, lab, t).assign(rule=f"source validation (Youden), source {src}", source=src))
thr = pd.concat(rows, ignore_index=True)
thr["train_hospital"] = thr.labeler.map(A.train_hospital)
thr["source"] = thr["source"].fillna(thr.train_hospital) if "source" in thr else thr.train_hospital
thr.to_csv(T / "threshold_transport.csv", index=False)

# 4. hospital-specific calibration
cal = A.hospital_calibration(W, ZS + SUP1 + POOLED)
cal.to_csv(T / "calibration_by_hospital.csv", index=False)

json.dump({"range": rng, "external_paired": ext.to_dict("records"), "threshold_transport": thr.to_dict("records"),
           "calibration_by_hospital": cal.to_dict("records")}, open(OUT / "revision_digest.json", "w"), indent=1, default=float)
print("revision tables written:", len(ext), "paired comparisons;", thr.rule.value_counts().to_dict())
