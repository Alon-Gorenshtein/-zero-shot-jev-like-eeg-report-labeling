import numpy as np
import pandas as pd
from eegrl import figures as F


def _prim():
    labs = ["kev-4b__full", "kev-4b__masked", "laya-en__full", "laya-en__masked", "gemma4-12b__full",
            "tfidf__full", "tfidf__masked", "rule__full", "nli-deberta__full"]
    rng = np.random.default_rng(0); a = rng.uniform(.8, .99, len(labs))
    return pd.DataFrame({"labeler": labs, "auroc": a, "auroc_lo": a - .01, "auroc_hi": a + .01,
                         "kappa": a - .1, "abstain": 0.0, "n": 1000})


def test_family_mapping():
    assert F.family("kev-9b__full") == "Decision model"
    assert F.family("laya-multi__masked") == "Decision model"
    assert F.family("gemma4-e4b__full") == "General LLM"
    assert F.family("tfidf-train-I0003__full") == "Supervised"
    assert F.family("rule__full") == "Rule"


def test_figures_render(tmp_path):
    prim = _prim()
    F.fig_forest(prim, tmp_path)
    F.fig_masked(prim, tmp_path)
    sub = pd.DataFrame([{"stratum_type": st, "stratum": s, "labeler": l, "n": 100, "auroc": .9, "lo": .88, "hi": .92, "kappa": .8}
                        for st, ss in [("hospital", ["I0001-S0001", "I0003"]), ("era", ["2000s", "2010s"])]
                        for s in ss for l in ["kev-4b__full", "tfidf__full"]])
    F.fig_heatmap(sub, tmp_path)
    for stem in ["Fig3_agreement_forest", "Fig4_masked_vs_full", "eFig_subgroup_heatmap"]:
        assert (tmp_path / f"{stem}.pdf").exists() and (tmp_path / f"{stem}.png").exists()


def test_transfer_figure_renders(tmp_path):
    rows = []
    for lab in ["tfidf-train-H1__full", "tfidf-train-H2__full", "kev-4b__full", "tfidf__full"]:
        for h in ["H1", "H2"]:
            rows.append({"labeler": lab, "hospital": h, "auroc": .9, "lo": .88, "hi": .92})
    gaps = pd.DataFrame([{"labeler": l, "kind": "x", "gap": .05, "lo": .02, "hi": .08} for l in {r["labeler"] for r in rows}])
    F.fig_transfer(pd.DataFrame(rows), gaps, tmp_path)
    assert (tmp_path / "Fig2_hospital_transfer.png").exists()


def test_pretty_names_for_variants():
    assert F.pretty("laya-en-512__full") == "Laya (English, 512 tokens)"
    assert F.pretty("kev-4b-findings__full") == "Kev-4B"


def test_forest_marks_pooled_upper_bound_and_subset(tmp_path):
    prim = _prim()
    prim = pd.concat([prim, prim.iloc[[0]].assign(labeler="kev-27b__full")], ignore_index=True)
    fig = F.fig_forest(prim, tmp_path)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("pooled-data comparator" in s for s in labels)
    assert any(s.startswith("Kev-27B") and "†" in s for s in labels)


def test_masked_figure_has_delta_panel(tmp_path):
    fig = F.fig_masked(_prim(), tmp_path)
    assert len(fig.axes) == 2
    assert (tmp_path / "Fig4_masked_vs_full.png").exists()


def test_external_principal_restricts_panel_b(tmp_path):
    rows = [{"labeler": l, "hospital": h, "auroc": .9, "lo": .88, "hi": .92}
            for l in ["kev-4b__full", "kev-9b__full", "nli-deberta__full", "tfidf-train-H1__full"] for h in ["H1", "H2"]]
    ext = pd.DataFrame([{"zero_shot": z, "supervised": "tfidf-train-H1__full", "target_hospital": "H2", "n": 100,
                         "diff": .01, "lo": -.01, "hi": .03, "p_boot": .5, "q_bh": .6}
                        for z in ["kev-4b__full", "kev-9b__full", "nli-deberta__full", "laya-en__full"]])
    fig = F.fig_external(pd.DataFrame(rows), ext, tmp_path, principal=["kev-4b__full", "kev-9b__full", "nli-deberta__full"])
    leg = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert leg == ["Kev-4B", "Kev-9B", "DeBERTa-v3 NLI"]


def test_masked_figure_orders_models_by_size_within_family(tmp_path):
    prim = _prim()
    extra = []
    for b in ["kev-27b", "kev-9b"]:
        for v in ["full", "masked"]:
            extra.append(prim.iloc[0].copy()); extra[-1]["labeler"] = f"{b}__{v}"
    prim = pd.concat([prim, pd.DataFrame(extra)], ignore_index=True)
    fig = F.fig_masked(prim, tmp_path)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    kev = [s for s in labels if s.startswith("Kev")]
    assert kev == ["Kev-4B", "Kev-9B", "Kev-27B"]


def _masked_subset_prim():
    prim = pd.DataFrame({"labeler": ["qwen38-27b__full", "qwen38-27b__masked", "kev-4b__full", "kev-4b__masked"],
                         "auroc": [.94, .90, .96, .90], "n": [20000, 6000, 20000, 20000]})
    sub = pd.DataFrame({"labeler": ["qwen38-27b__full"], "auroc": [.93], "n": [6000]})
    return prim, sub


def test_masked_delta_uses_same_reports_when_masked_ran_on_a_subset(tmp_path):
    prim, sub = _masked_subset_prim()
    fig = F.fig_masked(prim, tmp_path, subset=sub)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    widths = dict(zip(labels, [p.get_width() for p in fig.axes[1].patches]))
    assert set(labels) == {"Kev-4B", "Qwen3.8-27B†"}
    assert abs(widths["Qwen3.8-27B†"] - (-.03)) < 1e-9
    assert abs(widths["Kev-4B"] - (-.06)) < 1e-9


def test_masked_delta_refuses_mismatched_report_sets_without_subset(tmp_path):
    prim, _ = _masked_subset_prim()
    try:
        F.fig_masked(prim, tmp_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "qwen38-27b" in str(e)


def test_external_comparators_restrict_panel_b_rows(tmp_path):
    rows = [{"labeler": l, "hospital": h, "auroc": .9, "lo": .88, "hi": .92}
            for l in ["kev-4b__full", "tfidf-train-H1__full", "clinicalbert-train-H1__full"] for h in ["H1", "H2"]]
    ext = pd.DataFrame([{"zero_shot": "kev-4b__full", "supervised": s, "target_hospital": "H2", "n": 100,
                         "diff": .01, "lo": -.01, "hi": .03, "p_boot": .5, "q_bh": .6}
                        for s in ["tfidf-train-H1__full", "clinicalbert-train-H1__full"]])
    fig = F.fig_external(pd.DataFrame(rows), ext, tmp_path, principal=["kev-4b__full"], comparators=("tfidf",))
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert len(labels) == 1 and labels[0].startswith("TF-IDF")
