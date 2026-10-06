import numpy as np
import pandas as pd
from eegrl import analyze as A


def _W(n=600):
    rng = np.random.default_rng(0); y = rng.integers(0, 2, n)
    return pd.DataFrame({"report_id": [f"r{i}" for i in range(n)], "patient_id": np.arange(n) // 2, "label": y,
        "site": rng.choice(["I0001", "I0003"], n), "hospital": rng.choice(["I0001-S0001", "I0001-S0002", "I0003"], n),
        "age": rng.integers(1, 90, n), "year": rng.integers(1996, 2025, n),
        "n_chars": rng.integers(500, 5000, n), "f_spikes": rng.integers(0, 2, n),
        "p__good__full": np.clip(y + rng.normal(0, .3, n), 0, 1), "p__bad__full": rng.random(n)})


def test_primary_table_rows_and_ci_order():
    t = A.primary_table(_W(), B=100)
    assert set(t.labeler) == {"good__full", "bad__full"}
    g = t.set_index("labeler").loc["good__full"]
    assert g.auroc_lo <= g.auroc <= g.auroc_hi and g.auroc > .9


def test_comparisons_have_bh_q():
    c = A.comparisons(_W(), ref="good__full", B=100)
    assert {"diff", "lo", "hi", "p_boot", "q_bh"} <= set(c.columns)


def test_subgroups_include_age_group_and_era():
    s = A.subgroups(_W(), ["good__full"], B=50)
    assert {"hospital", "age_group", "era"} <= set(s.stratum_type)


def _W3(n=900):
    rng = np.random.default_rng(1); y = rng.integers(0, 2, n)
    hosp = np.repeat(["H1", "H2", "H3"], n // 3)
    noise = np.where(hosp == "H1", .2, 1.2)          # "trained on H1": good on H1, poor elsewhere
    return pd.DataFrame({"report_id": [f"r{i}" for i in range(n)], "patient_id": np.arange(n), "label": y, "hospital": hosp,
                         "p__tfidf-train-H1__full": y + rng.normal(0, 1, n) * noise,
                         "p__kev-4b__full": y + rng.normal(0, .4, n)})


def test_train_hospital_parsing():
    assert A.train_hospital("tfidf-train-I0001-S0002__full") == "I0001-S0002"
    assert A.train_hospital("clinicalbert-train-I0003__full") == "I0003"
    assert A.train_hospital("kev-4b__full") is None


def test_generalization_gap_supervised_positive_zero_shot_small():
    W = _W3()
    g_sup = A.generalization_gap(W, "tfidf-train-H1__full", B=100)
    g_zs = A.generalization_gap(W, "kev-4b__full", B=100)
    assert g_sup["kind"] == "in_minus_out" and g_sup["gap"] > .15 and g_sup["lo"] > 0
    assert g_zs["kind"] == "max_minus_min" and g_zs["gap"] < .08
    assert g_sup["lo"] <= g_sup["gap"] <= g_sup["hi"]


def test_transfer_table_marks_in_and_out():
    t = A.transfer_table(_W3(), ["tfidf-train-H1__full", "kev-4b__full"], B=50)
    r = t.set_index(["labeler", "hospital"])
    assert r.loc[("tfidf-train-H1__full", "H1"), "role"] == "in-hospital"
    assert r.loc[("tfidf-train-H1__full", "H2"), "role"] == "out-of-hospital"
    assert r.loc[("kev-4b__full", "H3"), "role"] == "zero-shot"


def test_abnormality_category_hierarchy():
    base = {f"fr_{f}": 0 for f in ["seizure", "status", "lpd", "gpd", "lrda", "grda", "bipd", "bs", "spikes",
                                   "foc slowing", "gen slowing"]}
    rows = [dict(base, label=1, fr_seizure=1, fr_spikes=1), dict(base, label=1, fr_lpd=1), dict(base, label=1, fr_bs=1),
            dict(base, label=1, fr_spikes=1), dict(base, label=1, **{"fr_foc slowing": 1}),
            dict(base, label=1, **{"fr_gen slowing": 1}), dict(base, label=1), dict(base, label=0)]
    cats = A.abnormality_category(pd.DataFrame(rows)).tolist()
    assert cats == ["Seizure or status", "Periodic or rhythmic pattern", "Burst suppression", "Epileptiform discharges",
                    "Focal slowing only", "Generalized slowing only", "Other abnormality", "Normal"]


def test_detection_by_category_rates():
    W = pd.DataFrame({"label": [1, 1, 1, 0], "patient_id": [1, 2, 3, 4], "hospital": ["H"] * 4,
                      "category": ["A", "A", "B", "Normal"], "p__m__full": [.9, .2, .8, .1]})
    t = A.detection_by_category(W, ["m__full"], B=20).set_index("category")
    assert t.loc["A", "rate"] == .5 and t.loc["B", "rate"] == 1.0 and t.loc["Normal", "rate"] == 1.0


def test_findings_table_uses_report_and_annotation_refs():
    rng = np.random.default_rng(0); n = 400; y = rng.integers(0, 2, n)
    W = pd.DataFrame({"patient_id": np.arange(n), "hospital": np.repeat(["H1", "H2"], n // 2),
                      "fr_lpd": y, "fa_lpd": y, "p__m__lpd": y * .8 + rng.random(n) * .3})
    t = A.findings_table(W, ["m"], ["lpd"], B=20)
    assert set(t.reference) == {"report", "annotation"} and (t.auroc > .9).all()


def test_load_predictions_skips_findings_files(tmp_path, monkeypatch):
    c = pd.DataFrame({"report_id": ["a", "b"], "split": ["test", "test"], "text": ["", ""], "label": [0, 1]})
    (tmp_path / "inter").mkdir(); (tmp_path / "pred").mkdir()
    c.to_parquet(tmp_path / "inter" / "cohort.parquet")
    pd.DataFrame({"report_id": ["a", "b"], "p_abnormal": [.1, .9]}).to_parquet(tmp_path / "pred" / "m__full.parquet")
    pd.DataFrame({"report_id": ["a", "b"], "p_lpd": [.1, .9]}).to_parquet(tmp_path / "pred" / "m-findings__full.parquet")
    monkeypatch.setattr(A, "INTER", tmp_path / "inter"); monkeypatch.setattr(A, "PRED", tmp_path / "pred")
    W = A.load_predictions("test")
    assert "p__m__full" in W and not any("findings" in col for col in W.columns)


def test_generalization_gap_on_kappa():
    W = _W3()
    g = A.generalization_gap(W, "tfidf-train-H1__full", B=50, stat="kappa")
    assert g["stat"] == "kappa" and g["gap"] > 0


def test_verified_sensitivity():
    W = pd.DataFrame({"patient_id": range(6), "hospital": ["H"] * 6, "fv_lpd": [1, 1, 1, 1, 0, 0],
                      "p__m__lpd": [.9, .8, .2, .7, .9, .1]})
    t = A.verified_sensitivity(W, ["m"], ["lpd"], B=20)
    r = t[t.hospital == "All"].iloc[0]
    assert r.n_verified == 4 and abs(r.sensitivity - .75) < 1e-9


def test_latent_class_findings_runs_per_finding():
    rng = np.random.default_rng(0); n = 3000; z = (rng.random(n) < .2).astype(int)
    noisy = lambda s, c: np.where(z == 1, rng.random(n) < s, rng.random(n) > c).astype(float)
    W = pd.DataFrame({"patient_id": np.arange(n), "fr_lpd": noisy(.95, .7), "fa_lpd": noisy(.6, .95),
                      "p__m__lpd": noisy(.9, .95), "p__k__lpd": noisy(.85, .97)})
    r = A.latent_class_findings(W, ["m", "k"], ["lpd"], B=10)
    ref = r[(r.finding == "lpd") & (r.source == "report reference")].iloc[0]
    assert abs(ref.spec - .7) < .05 and abs(ref.sens - .95) < .05


def test_comparisons_use_common_reports_only():
    W = _W()
    W.loc[W.index[:300], "p__bad__full"] = np.nan   # subset model
    c = A.comparisons(W, ref="good__full", B=50)
    assert int(c.iloc[0].n_common) == 300


def test_restrict_to_subset_blanks_other_reports():
    W = pd.DataFrame({"report_id": ["a", "b", "c"], "p__kev-27b__full": [.1, .2, .3], "p__kev-4b__full": [.1, .2, .3]})
    out = A.restrict_to_subset(W, {"a", "c"}, prefixes=("kev-27b",))
    assert out["p__kev-27b__full"].isna().tolist() == [False, True, False]
    assert out["p__kev-4b__full"].notna().all()


def test_consensus_audit_counts_unanimous_disagreement_by_hospital():
    W = pd.DataFrame({"hospital": ["A"] * 4 + ["B"] * 4, "label": [0, 0, 1, 1] * 2,
                      "p__m1__full": [.9, .1, .1, .9, .9, .9, .9, .9],
                      "p__m2__full": [.8, .9, .2, .8, .2, .9, .9, .9]})
    r = A.consensus_audit(W, ["m1__full", "m2__full"]).set_index(["hospital", "reference"])
    assert r.loc[("A", "normal"), "n"] == 2 and r.loc[("A", "normal"), "k"] == 1
    assert r.loc[("A", "abnormal"), "k"] == 1          # both models call normal
    assert r.loc[("B", "normal"), "k"] == 1 and r.loc[("B", "abnormal"), "k"] == 0
    assert (r.lo <= r.prop).all() and (r.prop <= r.hi).all()


def test_generalization_gap_range_applies_max_minus_min_to_trained_models():
    W = _W3()
    native = A.generalization_gap(W, "tfidf-train-H1__full", B=50)
    rng_ = A.generalization_gap(W, "tfidf-train-H1__full", B=50, kind="range")
    assert native["kind"] == "in_minus_out" and rng_["kind"] == "max_minus_min"
    assert rng_["gap"] >= native["gap"] - 1e-9          # max-min >= in minus mean(out) when in-hospital is the max


def test_external_paired_compares_on_same_external_reports():
    W = _W3()
    r = A.external_paired(W, ["kev-4b__full"], ["tfidf-train-H1__full"], B=50)
    assert set(r.target_hospital) == {"H2", "H3"}          # never the comparator's training hospital
    assert {"diff", "lo", "hi", "p_boot", "q_bh", "n"} <= set(r.columns)
    assert (r["diff"] > 0).all()                           # kev beats the poorly transported model out of hospital


def test_val_threshold_youden_and_transport():
    rng = np.random.default_rng(3); n = 400; y = rng.integers(0, 2, n)
    val = pd.DataFrame({"label": y, "hospital": "H1", "p__m__full": y * .3 + .4 + rng.normal(0, .05, n)})
    t = A.youden_threshold(val.label.values, val["p__m__full"].values)
    assert .45 < t < .65                                    # separates 0.4 from 0.7, unlike a fixed 0.5 would not matter here
    W = _W3()
    out = A.transported_threshold(W, "kev-4b__full", .5)
    assert set(out.hospital) == {"H1", "H2", "H3"} and {"sens", "spec", "kappa", "threshold"} <= set(out.columns)


def test_hospital_calibration_slope_near_one_for_calibrated_probs():
    rng = np.random.default_rng(4); n = 6000; p = rng.uniform(.02, .98, n); y = (rng.random(n) < p).astype(int)
    W = pd.DataFrame({"label": y, "hospital": np.repeat(["H1", "H2"], n // 2), "p__m__full": p})
    c = A.hospital_calibration(W, ["m__full"]).set_index("hospital")
    assert abs(c.loc["H1", "slope"] - 1) < .15 and abs(c.loc["H1", "intercept"]) < .15
    assert {"brier", "ece", "n"} <= set(c.columns)


def test_hospital_calibration_saturated_probabilities_do_not_explode():
    y = np.array([0] * 50 + [1] * 50)
    W = pd.DataFrame({"label": y, "hospital": "H1", "p__m__full": np.r_[np.ones(50), np.ones(50)]})  # all predicted 1
    c = A.hospital_calibration(W, ["m__full"]).iloc[0]
    assert np.isnan(c.intercept) or abs(c.intercept) < 50


def test_wilson_handles_zero_and_bounds():
    p, lo, hi = A.wilson(0, 0)
    assert np.isnan(p) and np.isnan(lo) and np.isnan(hi)
    p, lo, hi = A.wilson(5, 10)
    assert p == .5 and 0 < lo < .5 < hi < 1


def test_mention_annotation_check_groups_by_reference_and_mention():
    C = pd.DataFrame({"text": ["no seizures", "two seizures were recorded", "no seizures", "seizure"],
                      "hospital": ["H1", "H1", "H2", "H2"],
                      "fr_seizure": [1, 1, 1, 0], "fa_seizure": [0, 1, 0, 0]})
    cls = lambda t, f: "negated only" if t.startswith("no") else ("affirmed" if "recorded" in t else "mention without verdict")
    r = A.mention_annotation_check(C, "seizure", cls)
    a = r[(r.hospital == "All") & (r.reference == 1)].set_index("mention")
    assert a.loc["negated only", "n"] == 2 and a.loc["negated only", "annotated"] == 0
    assert a.loc["affirmed", "prop"] == 1.0
    assert set(r.hospital) == {"All", "H1", "H2"}
    assert {"finding", "lo", "hi"} <= set(r.columns)


def test_consensus_annotation_check_splits_reference_normals():
    W = pd.DataFrame({"hospital": ["A"] * 5, "label": [0, 0, 0, 0, 1],
                      "p__m1__full": [.9, .9, .1, .2, .9], "p__m2__full": [.8, .9, .9, .1, .9],
                      "ann": [1, 0, 0, 0, 1]})
    r = A.consensus_annotation_check(W, ["m1__full", "m2__full"], "ann").set_index("group")
    assert r.loc["concordant abnormal call", "n"] == 2 and r.loc["concordant abnormal call", "annotated"] == 1
    assert r.loc["other reference-normal", "n"] == 2 and r.loc["other reference-normal", "annotated"] == 0


def test_source_thresholds_per_hospital_and_empty_source_raises():
    rng = np.random.default_rng(1); n = 300; y = rng.integers(0, 2, n)
    V = pd.DataFrame({"label": y, "hospital": np.where(np.arange(n) < 150, "H1", "H2"),
                      "p__m__full": y * .4 + .3 + rng.normal(0, .03, n)})
    t = A.source_thresholds(V, "m__full", ["H1", "H2"])
    assert set(t) == {"H1", "H2"} and all(.35 < v < .75 for v in t.values())
    V.loc[V.hospital == "H2", "p__m__full"] = np.nan
    try:
        A.source_thresholds(V, "m__full", ["H1", "H2"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "H2" in str(e)


def test_prediction_agreement_on_common_reports():
    a = pd.DataFrame({"report_id": ["r1", "r2", "r3"], "p_abnormal": [.1, .5, .9]})
    b = pd.DataFrame({"report_id": ["r2", "r3", "r4"], "p_abnormal": [.52, .9, .3]})
    d = A.prediction_agreement(a, b)
    assert d["n"] == 2 and abs(d["max_abs_diff"] - .02) < 1e-9 and d["pearson_r"] > .99


def test_source_thresholds_raises_on_partly_missing_scores():
    rng = np.random.default_rng(2); n = 300; y = rng.integers(0, 2, n)
    V = pd.DataFrame({"label": y, "hospital": np.where(np.arange(n) < 150, "H1", "H2"),
                      "p__m__full": y * .4 + .3 + rng.normal(0, .03, n)})
    V.loc[3, "p__m__full"] = np.nan
    try:
        A.source_thresholds(V, "m__full", ["H1", "H2"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "H1" in str(e) and "missing" in str(e)


def _mention_fixture():
    C = pd.DataFrame({"report_id": ["a", "b", "c", "d", "e"],
                      "text": ["neg", "neg", "aff", "aff", "neg"],
                      "fr_seizure": [1, 1, 1, 1, 0]})
    classify = lambda t, f: {"neg": "negated only", "aff": "affirmed"}[t]
    return C, classify


def test_mention_model_check_counts_each_labeler_on_its_own_reports():
    C, classify = _mention_fixture()
    preds = {"m1": pd.DataFrame({"report_id": ["a", "b", "c", "d", "e"], "p_seizure": [.9, .1, .8, .7, .2]}),
             "m2": pd.DataFrame({"report_id": ["a", "c"], "p_seizure": [.1, .9]})}
    out = A.mention_model_check(C, "seizure", classify, preds)
    r = out.set_index(["labeler", "reference", "mention"])
    assert r.loc[("m1", 1, "negated only"), "n"] == 2 and r.loc[("m1", 1, "negated only"), "called_present"] == 1
    assert r.loc[("m1", 1, "affirmed"), "prop"] == 1.0
    assert r.loc[("m2", 1, "negated only"), "n"] == 1 and r.loc[("m2", 1, "negated only"), "prop"] == 0.0
    assert ("m2", 0, "negated only") not in r.index  # m2 never scored report e
    assert {"finding", "lo", "hi"} <= set(out.columns)


def test_mention_model_check_raises_on_missing_scores():
    C, classify = _mention_fixture()
    preds = {"m1": pd.DataFrame({"report_id": ["a", "b"], "p_seizure": [.9, np.nan]})}
    try:
        A.mention_model_check(C, "seizure", classify, preds)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "m1" in str(e) and "missing" in str(e)
