import json
import numpy as np
import pandas as pd
from eegrl import analyze as A
from eegrl.config import OUT, INTER

T = OUT / "tables"; T.mkdir(exist_ok=True, parents=True)
W = A.load_predictions("test")
from eegrl.speed import bench_sample
SUB = set(bench_sample(pd.read_parquet(INTER / "cohort.parquet", columns=["report_id", "split", "hospital"]), 6000).report_id)
W = A.restrict_to_subset(W, SUB)
prim = A.primary_table(W); prim.to_csv(T / "primary_agreement.csv", index=False)
full = [l for l in prim.labeler if l.endswith("__full") and not l.startswith("rule")]
dec = prim[prim.labeler.str.match(r"(kev|laya)") & prim.labeler.str.endswith("__full")]
best_decision = dec.iloc[0].labeler
keep = [c for c in W.columns if not c.startswith("p__") or c.endswith("__full")]
comp = A.comparisons(W[keep], ref=best_decision); comp.to_csv(T / "comparisons_vs_best_decision.csv", index=False)
masked = prim[prim.labeler.str.endswith("__masked")]; masked.to_csv(T / "masked_agreement.csv", index=False)
sub = A.subgroups(W, full + ["rule__full"]); sub.to_csv(T / "subgroups.csv", index=False)
cal = A.calibration(W, [l for l in prim.labeler if not l.startswith("rule")]); cal.to_csv(T / "calibration.csv", index=False)
fams = [r"(gemma|qwen)", r"nli", r"tfidf__", r"clinicalbert"]
one_per_family = [best_decision] + [prim[prim.labeler.str.match(f) & prim.labeler.str.endswith("__full")].iloc[0].labeler
                                    for f in fams if (prim.labeler.str.match(f) & prim.labeler.str.endswith("__full")).any()]
# latent class: one main configuration per model (no paraphrase / 512-token duplicates, no per-hospital-trained copies)
lc_raters = [l for l in full if not any(x in l for x in ("-para", "-512", "-train-"))]
lc_all, lc_fam = A.latent_class(W, lc_raters), A.latent_class(W, one_per_family)
disc = A.discordance_model(W, best_decision); disc.to_csv(T / "discordance_or.csv", index=False)
tax = A.disagreement_taxonomy(W, best_decision); tax.to_csv(T / "disagreement_taxonomy.csv", index=False)
gen_labs = [l for l in prim.labeler if l.endswith("__full") and "-para" not in l]
xfer = A.transfer_table(W, gen_labs); xfer.to_csv(T / "transfer_matrix.csv", index=False)
gaps = pd.DataFrame([A.generalization_gap(W, l) for l in gen_labs]); gaps.to_csv(T / "generalization_gap.csv", index=False)
# key secondary (post hoc, disclosed): Cohen kappa at the transported 0.5 decision rule, no local recalibration
gaps_k = pd.DataFrame([A.generalization_gap(W, l, stat="kappa") for l in gen_labs]); gaps_k.to_csv(T / "generalization_gap_kappa.csv", index=False)
# abnormality type (hierarchical, report-sourced findings) and benign-variant normal reports
from eegrl.config import BENIGN_VARIANTS, TARGET_FINDINGS
W["category"] = A.abnormality_category(W)
cat = A.detection_by_category(W, gen_labs); cat.to_csv(T / "detection_by_abnormality_type.csv", index=False)
catH = pd.concat([A.detection_by_category(g, gen_labs).assign(hospital=h) for h, g in W.groupby("hospital")])
catH.to_csv(T / "detection_by_abnormality_type_hospital.csv", index=False)
Nn = W[W.label == 0].copy()
Nn["benign_variant"] = np.where(Nn[[f"fr_{b}" for b in BENIGN_VARIANTS]].sum(axis=1) > 0, "Benign variant or sleep transient mentioned", "None mentioned")
ben = A.detection_by_category(Nn, gen_labs, by="benign_variant"); ben.to_csv(T / "specificity_benign_variants.csv", index=False)
# independent check: recording annotations (technologist/physician), not derived from the report
ann = ["spikes", "seizure", "lpd", "gpd", "lrda", "grda", "bs"]
W["annotated_abnormal"] = np.where(W[[f"fa_{f}" for f in ann]].sum(axis=1) > 0, "Annotated epileptiform, periodic or seizure", "No such annotation")
annt = A.detection_by_category(W, gen_labs, by="annotated_abnormal"); annt.to_csv(T / "annotation_check.csv", index=False)
WF, fnames = A.load_findings_predictions("test")
ftab = A.findings_table(WF, fnames, TARGET_FINDINGS) if fnames else pd.DataFrame()
ftab.to_csv(T / "findings_agreement.csv", index=False)
vsen = A.verified_sensitivity(WF, fnames, TARGET_FINDINGS) if fnames else pd.DataFrame()
vsen.to_csv(T / "findings_verified_sensitivity.csv", index=False)
# reference-label audit (post hoc, disclosed): how each finding is mentioned in reports the reference calls positive
from eegrl.text import mention_class
C = pd.read_parquet(INTER / "cohort.parquet"); C = C[C.split == "test"]
aud = []
for f in TARGET_FINDINGS:
    for ref_val, g in C.groupby(f"fr_{f}"):
        for h, gh in [("All", g)] + list(g.groupby("hospital")):
            vc = gh.text.map(lambda t: mention_class(t, f)).value_counts(normalize=True)
            aud += [{"finding": f, "reference": int(ref_val), "hospital": h, "n": len(gh), "mention": k, "share": v} for k, v in vc.items()]
audit = pd.DataFrame(aud); audit.to_csv(T / "reference_audit.csv", index=False)
lcf = A.latent_class_findings(WF, fnames, TARGET_FINDINGS) if fnames else pd.DataFrame()
lcf.to_csv(T / "latent_class_findings.csv", index=False)
CONSENSUS = ["kev-4b__full", "kev-9b__full", "nli-deberta__full", "laya-en__full"]  # 3 zero-shot model families (not statistically independent), full test split
cons = A.consensus_audit(W, CONSENSUS); cons.to_csv(T / "reference_consensus_audit.csv", index=False)
# non-text corroboration (post hoc): recording annotations by mention class, and for concordant zero-shot calls
mann = pd.concat([A.mention_annotation_check(C, f, mention_class) for f in TARGET_FINDINGS], ignore_index=True)
mann.to_csv(T / "mention_annotation.csv", index=False)
cann = A.consensus_annotation_check(W.assign(annotated_any=(W.annotated_abnormal != "No such annotation").astype(int)),
                                    CONSENSUS, "annotated_any")
cann.to_csv(T / "consensus_annotation.csv", index=False)
# like-for-like with Kev-27B: every labeler on the fixed 6,000-report subset
sub_ids = set(pd.read_csv(T / "subset6000_report_ids.csv").report_id)
prim_sub = A.primary_table(W[W.report_id.isin(sub_ids)])
prim_sub.to_csv(T / "primary_agreement_subset6000.csv", index=False)
coh = W.groupby("hospital").agg(n=("label", "size"), abnormal=("label", "sum"), patients=("patient_id", "nunique"))
json.dump({"n_test": len(W), "best_decision_model": best_decision, "cohort_test": coh.reset_index().to_dict("records"),
           "primary": prim.to_dict("records"), "transfer": xfer.to_dict("records"), "by_abnormality_type": cat.to_dict("records"),
           "benign_variants": ben.to_dict("records"), "annotation_check": annt.to_dict("records"),
           "findings": ftab.to_dict("records"), "findings_verified_sensitivity": vsen.to_dict("records"),
           "reference_audit": audit.to_dict("records"), "reference_consensus_audit": cons.to_dict("records"), "mention_annotation": mann.to_dict("records"), "consensus_annotation": cann.to_dict("records"), "primary_subset6000": prim_sub.to_dict("records"), "latent_class_findings": lcf.to_dict("records"), "generalization_gap": gaps.to_dict("records"), "generalization_gap_kappa": gaps_k.to_dict("records"), "masked": masked.to_dict("records")},
          open(OUT / "results_digest.json", "w"), indent=1, default=float)
json.dump({"comparisons": comp.to_dict("records"), "latent_class_all": lc_all, "latent_class_one_per_family": lc_fam,
           "calibration": cal.to_dict("records"), "discordance": disc.to_dict("records")},
          open(OUT / "stats_digest.json", "w"), indent=1, default=float)
print(prim.head(30).to_string())
