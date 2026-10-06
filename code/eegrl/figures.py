import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
LABEL, MUTED, GRID, REF = "#2B2B2B", "#6B7280", "#F3F4F6", "#9CA3AF"
FAMILY_COLOR = {"Decision model": "#00468B", "General LLM": "#D97848", "Zero-shot NLI": "#7E6BAF",
                "Supervised": "#ADB6B6", "Rule": "#1F2A44"}
FAMILY_ORDER = list(FAMILY_COLOR)

PRETTY = {"kev-0.8b": "Kev-0.8B", "kev-4b": "Kev-4B", "kev-9b": "Kev-9B", "kev-27b": "Kev-27B",
          "laya-en": "Laya (English)", "laya-multi": "Laya (multilingual)", "laya-en-512": "Laya (English, 512 tokens)",
          "gemma4-e4b": "Gemma-4-E4B", "gemma4-12b": "Gemma-4-12B", "qwen38-27b": "Qwen3.8-27B",
          "nli-deberta": "DeBERTa-v3 NLI", "tfidf": "TF-IDF + LR", "clinicalbert": "Bio_ClinicalBERT",
          "rule": "Verdict regex"}


def family(labeler: str) -> str:
    base = labeler.split("__")[0]
    if base.startswith(("kev", "laya")):
        return "Decision model"
    if base.startswith(("gemma", "qwen")):
        return "General LLM"
    if base.startswith("nli"):
        return "Zero-shot NLI"
    if base.startswith("rule"):
        return "Rule"
    return "Supervised"


def pretty(labeler: str) -> str:
    base = labeler.split("__")[0].replace("-findings", "")
    para = base.endswith("-para")
    base = base.removesuffix("-para")
    m = re.match(r"(tfidf|clinicalbert)-train-(.+)", base)
    name = f"{PRETTY[m.group(1)]} (trained on {m.group(2)})" if m else PRETTY.get(base, base)
    return name + (", paraphrase" if para else "")


def _save(fig, out, stem):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _style(ax):
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(REF); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=LABEL, labelsize=9)


def fig_forest(prim: pd.DataFrame, out, stem="Fig3_agreement_forest", subset_only=("kev-27b",)):
    """AUROC (95% CI) vs the reference label, full-report input, grouped by model family."""
    d = prim[prim.labeler.str.endswith("__full") & prim.auroc.notna()].copy()
    d["family"] = d.labeler.map(family)
    d["fo"] = d.family.map(FAMILY_ORDER.index)
    d = d.sort_values(["fo", "auroc"], ascending=[True, False])
    rows, y, ylabels, headers = [], 0, [], []
    for fam, g in d.groupby("fo", sort=True):
        headers.append((y, FAMILY_ORDER[fam])); y += 1
        for _, r in g.iterrows():
            name = pretty(r.labeler)
            if r.labeler in ("tfidf__full", "clinicalbert__full"):
                name += " (pooled-data comparator)"
            if r.labeler.split("__")[0] in subset_only:
                name += "\u2020"
            rows.append((y, r)); ylabels.append((y, name)); y += 1
        y += 0.5
    fig, ax = plt.subplots(figsize=(8, 0.32 * y + 1))
    for yy, r in rows:
        c = FAMILY_COLOR[family(r.labeler)]
        ax.plot([r.auroc_lo, r.auroc_hi], [yy, yy], color=c, lw=1.6, zorder=2)
        ax.scatter(r.auroc, yy, s=42, color=c, edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(1.005, yy, f"{r.auroc:.3f} ({r.auroc_lo:.3f}-{r.auroc_hi:.3f})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=LABEL)
    for yy, name in headers:
        ax.text(-0.01, yy, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=11, fontweight="bold", color=FAMILY_COLOR[name])
    ax.set_yticks([p for p, _ in ylabels]); ax.set_yticklabels([n for _, n in ylabels], fontsize=9)
    ax.invert_yaxis()
    lo = max(0.5, np.floor((d.auroc_lo.min() - 0.02) * 20) / 20)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("AUROC vs reference report label (95% CI)", fontsize=10, color=LABEL)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    _style(ax)
    _save(fig, out, stem)
    return fig


def fig_masked(prim: pd.DataFrame, out, stem="Fig4_masked_vs_full", subset=None):
    """a: AUROC with the full report vs with verdict sentences removed, ordered by family; b: change in AUROC.
    A labeler masked on fewer reports than its full run takes its full-report AUROC from `subset` (the same
    reports) and is marked with a dagger, so both panels compare like with like."""
    p = prim.assign(base=prim.labeler.str.split("__").str[0], variant=prim.labeler.str.split("__").str[1])
    w = p.pivot_table(index="base", columns="variant", values="auroc").dropna(subset=["full", "masked"])
    n = p.pivot_table(index="base", columns="variant", values="n")
    sub = {} if subset is None else subset.set_index("labeler")[["auroc", "n"]].to_dict("index")
    mark = set()
    for b in w.index:
        if n.at[b, "full"] != n.at[b, "masked"]:
            s = sub.get(f"{b}__full")
            if s is None or s["n"] != n.at[b, "masked"]:
                raise ValueError(f"{b}: full and masked AUROCs come from different report sets; pass a matching subset")
            w.at[b, "full"] = s["auroc"]; mark.add(b)
    w = w.assign(family=[family(b) for b in w.index])
    order = list(PRETTY)  # PRETTY lists models smallest to largest within each family
    w = w.assign(fo=w.family.map(FAMILY_ORDER.index), nm=[pretty(b) + ("\u2020" if b in mark else "") for b in w.index],
                 po=[order.index(b) if b in order else len(order) for b in w.index]).sort_values(["fo", "po", "nm"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 0.34 * len(w) + 1.2), sharey=True, gridspec_kw={"width_ratios": [1.4, 1]})
    for i, (b, r) in enumerate(w.iterrows()):
        c = FAMILY_COLOR[r.family]
        ax.plot([r.masked, r.full], [i, i], color="#D1D5DB", lw=1.4, zorder=1)
        ax.scatter(r.full, i, s=42, color=c, edgecolors="white", linewidths=0.8, zorder=3)
        ax.scatter(r.masked, i, s=42, facecolors="white", edgecolors=c, linewidths=1.4, zorder=3)
        bx.barh(i, r.masked - r.full, color=c, height=0.6, zorder=3)
    ax.set_yticks(range(len(w))); ax.set_yticklabels(list(w.nm), fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("AUROC vs reference report label", fontsize=10, color=LABEL)
    bx.axvline(0, color=REF, lw=0.8); bx.set_xlabel("\u0394AUROC, verdict-masked minus full", fontsize=10, color=LABEL)
    for a_ in (ax, bx):
        a_.spines["left"].set_visible(False); a_.tick_params(axis="y", length=0); _style(a_)
    h = [mlines.Line2D([], [], marker="o", ls="", color=LABEL, markeredgecolor="white", label="Full report"),
         mlines.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=LABEL, label="Verdict sentences removed")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.9 / max(len(w), 3) - 0.12), ncol=2, fontsize=9, frameon=False)
    for i, a_ in enumerate((ax, bx)):
        a_.text(-0.02 if i else -0.45, 1.02, "ab"[i], transform=a_.transAxes, fontsize=14, fontweight="bold", color=LABEL)
    fig.tight_layout()
    _save(fig, out, stem)
    return fig


def fig_heatmap(sub: pd.DataFrame, out, stem="eFig_subgroup_heatmap", value="kappa"):
    """Cohen kappa (reference vs model at 0.5) by stratum, one column per model."""
    d = sub[sub.stratum_type.isin(["hospital", "era", "age_group"])].copy()
    d["row"] = d.stratum_type.map({"hospital": "Hospital", "era": "Era", "age_group": "Age"}) + ": " + d.stratum.astype(str)
    order = {"Hospital": 0, "Age": 1, "Era": 2}
    rows = sorted(d.row.unique(), key=lambda r: (order[r.split(":")[0]], r))
    cols = sorted(d.labeler.unique(), key=lambda l: (FAMILY_ORDER.index(family(l)), l))
    M = d.pivot_table(index="row", columns="labeler", values=value).reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 0.38 * len(rows) + 1.4))
    im = ax.imshow(M.values, cmap="viridis", vmin=np.nanmin(M.values), vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < np.nanmean(M.values) else "#1F2A44")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([pretty(c) for c in cols], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Cohen κ vs reference label" if value == "kappa" else "AUROC", fontsize=9)
    cb.outline.set_visible(False)
    _save(fig, out, stem)


def fig_calibration(W: pd.DataFrame, labelers, out, stem="eFig_calibration", bins=10):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", color=REF, lw=0.7)
    for lab in labelers:
        p = W[f"p__{lab}"].values; ok = ~np.isnan(p); y = W.label.values[ok]; p = p[ok]
        idx = np.minimum((p * bins).astype(int), bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 20]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 20]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.2, color=FAMILY_COLOR[family(lab)], label=pretty(lab))
    ax.set_xlabel("Predicted probability of abnormal", fontsize=10); ax.set_ylabel("Observed proportion abnormal (reference)", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8, frameon=False)
    _style(ax)
    _save(fig, out, stem)


def fig_latent(lc: dict, out, stem="eFig_latent_class"):
    names = [k for k in lc if k != "prevalence"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.34 * len(names) + 1.2), sharey=True)
    for ax, key, lab in zip(axes, ["sens", "spec"], ["Estimated sensitivity", "Estimated specificity"]):
        for i, n in enumerate(names):
            c = "#1F2A44" if n == "reference" else FAMILY_COLOR[family(n)]
            lo, hi = lc[n][f"{key}_ci"]
            ax.plot([lo, hi], [i, i], color=c, lw=1.6); ax.scatter(lc[n][key], i, s=36, color=c, edgecolors="white", zorder=3)
        ax.set_xlabel(lab, fontsize=10); _style(ax)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(["Reference report label" if n == "reference" else pretty(n) for n in names], fontsize=9)
    axes[0].invert_yaxis()
    for i, ax in enumerate(axes):
        ax.text(-0.12 if i == 0 else -0.05, 1.03, "ab"[i], transform=ax.transAxes, fontsize=14, fontweight="bold", color=LABEL)
    _save(fig, out, stem)


def fig_transfer(xfer: pd.DataFrame, gaps: pd.DataFrame, out, stem="Fig2_hospital_transfer"):
    """a: AUROC by test hospital (columns) for each model (rows); in-hospital cells outlined.
    b: generalization gap (supervised: in minus mean out; zero-shot/pooled: max minus min), 95% CI."""
    from .analyze import train_hospital
    labs = sorted(xfer.labeler.unique(), key=lambda l: (FAMILY_ORDER.index(family(l)), train_hospital(l) or "", l))
    hosp = sorted(xfer.hospital.unique())
    M = xfer.pivot_table(index="labeler", columns="hospital", values="auroc").reindex(index=labs, columns=hosp)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.42 * len(labs) + 1.6), gridspec_kw={"width_ratios": [1.1, 1]})
    vmin = max(0.5, np.floor(np.nanmin(M.values) * 20) / 20)
    im = a.imshow(M.values, cmap="viridis", vmin=vmin, vmax=1.0, aspect="auto")
    for i, l in enumerate(labs):
        th = train_hospital(l)
        for j, h in enumerate(hosp):
            v = M.values[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                       color="white" if v < (vmin + 1) / 2 else "#1F2A44")
            if th and h in th.split("+"):
                a.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#FDAF91", lw=2.2))
    a.set_xticks(range(len(hosp))); a.set_xticklabels(hosp, fontsize=9)
    a.set_yticks(range(len(labs))); a.set_yticklabels([pretty(l) for l in labs], fontsize=9)
    a.set_xlabel("Test hospital", fontsize=10); a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02); cb.set_label("AUROC vs reference label", fontsize=9); cb.outline.set_visible(False)
    g = gaps.set_index("labeler").reindex(labs)
    for i, l in enumerate(labs):
        if l not in gaps.labeler.values:
            continue
        c = FAMILY_COLOR[family(l)]
        in_out = train_hospital(l) is not None
        b.plot([g.loc[l, "lo"], g.loc[l, "hi"]], [i, i], color=c, lw=1.6)
        b.scatter(g.loc[l, "gap"], i, s=44, marker="s" if in_out else "o", color=c if in_out else "white",
                  edgecolors=c, linewidths=1.4, zorder=3)
    h = [mlines.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=LABEL, label="Best minus worst hospital (zero-shot / pooled)"),
         mlines.Line2D([], [], marker="s", ls="", color=LABEL, label="Own minus other hospitals (trained on one hospital)")]
    b.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8.5, frameon=False)
    b.axvline(0, ls="--", color=REF, lw=0.7)
    b.set_ylim(len(labs) - .5, -.5); b.set_yticks([])
    b.set_xlabel("Generalization gap in AUROC (95% CI)", fontsize=10)
    _style(b); b.spines["left"].set_visible(False)
    for i, ax in enumerate((a, b)):
        ax.text(-0.02 if i else -0.55, 1.03, "ab"[i], transform=ax.transAxes, fontsize=14, fontweight="bold", color=LABEL)
    _save(fig, out, stem)


COST_OFFSETS = {("full", "nli-deberta"): (-100, 8), ("full", "laya-en"): (6, -12),
                ("masked", "gemma4-e4b"): (-70, -28), ("masked", "kev-4b"): (-55, 20), ("masked", "gemma4-12b"): (-25, -22),
                ("masked", "qwen38-27b"): (10, 24), ("masked", "kev-9b"): (30, -30), ("masked", "kev-27b"): (-20, -55)}


def fig_cost(prim: pd.DataFrame, speed: pd.DataFrame, out, stem="eFig_accuracy_vs_cost"):
    """AUROC (full and verdict-masked report) against inference cost, GPU-hours per 100,000 reports on one GB10."""
    sp = speed[speed.task == "binary"].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, variant, letter in zip(axes, ("full", "masked"), "ab"):
        for base, r in sp.iterrows():
            row = prim[prim.labeler == f"{base}__{variant}"]
            if row.empty or row.auroc.isna().all():
                continue
            row = row.iloc[0]; fam = family(base)
            ax.errorbar(r.hours_per_100k, row.auroc, yerr=[[row.auroc - row.auroc_lo], [row.auroc_hi - row.auroc]],
                        fmt="o", ms=7, color=FAMILY_COLOR[fam], mec="white", mew=0.8, elinewidth=0.8, zorder=3)
            off = COST_OFFSETS.get((variant, base))
            ax.annotate(pretty(base), (r.hours_per_100k, row.auroc), xytext=off or (5, 3), textcoords="offset points",
                        fontsize=7.5, color=LABEL,
                        arrowprops=dict(arrowstyle="-", color=REF, lw=0.6, shrinkA=0, shrinkB=4) if off and max(map(abs, off)) > 15 else None)
        ax.set_xscale("log"); _style(ax); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel("GPU-hours per 100,000 reports (log scale)", fontsize=10, color=LABEL)
        ax.set_ylabel("AUROC", fontsize=10, color=LABEL)
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", color=LABEL)
        ax.text(0.02, 0.02, "Full report" if variant == "full" else "Verdict-masked report", transform=ax.transAxes,
                fontsize=9, color=MUTED)
    handles = [mlines.Line2D([], [], marker="o", ls="", color=c, mec="white", ms=7, label=f)
               for f, c in FAMILY_COLOR.items() if f != "Rule"]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, out, stem)


EXT_ALL = [("kev-4b__full", "o"), ("kev-9b__full", "D"), ("kev-27b__full", "P"), ("laya-en__full", "s"), ("nli-deberta__full", "^")]
EXT_COLOR = {"kev-4b__full": "#00468B", "kev-9b__full": "#4F7FB8", "kev-27b__full": "#8FB3DE",
             "laya-en__full": "#1B7340", "nli-deberta__full": "#7E6BAF"}
EXT_ZS = EXT_ALL  # backward-compatible name


def fig_external(xfer: pd.DataFrame, ext: pd.DataFrame, out, stem="Fig2_external_hospitals", principal=None, comparators=None):
    """a: AUROC by test hospital for each labeler (in-hospital cells outlined).
    b: paired AUROC difference at each hospital not used for training, zero-shot model minus supervised model trained at another
    hospital, on the same reports (95% CI)."""
    from .analyze import train_hospital
    zs_list = [(z, mk) for z, mk in EXT_ALL if principal is None or z in principal]
    order = list(PRETTY)
    labs = sorted(xfer.labeler.unique(), key=lambda l: (FAMILY_ORDER.index(family(l)), train_hospital(l) or "",
                                                       order.index(re.sub(r"-train-.*", "", l.split("__")[0])) if re.sub(r"-train-.*", "", l.split("__")[0]) in order else 99))
    hosp = sorted(xfer.hospital.unique())
    M = xfer.pivot_table(index="labeler", columns="hospital", values="auroc").reindex(index=labs, columns=hosp)
    comps = ext[["supervised", "target_hospital"]].drop_duplicates()
    if comparators:  # e.g. ("tfidf",): panel b shows only these supervised comparators
        comps = comps[comps.supervised.map(lambda s: s.startswith(tuple(comparators)))]
    comps = comps.assign(k=comps.supervised.map(lambda s: ("tfidf" not in s, train_hospital(s)))).sort_values(["k", "target_hospital"])
    rows = list(comps[["supervised", "target_hospital"]].itertuples(index=False))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, max(0.42 * len(labs), 0.55 * len(rows)) + 1.8), gridspec_kw={"width_ratios": [1, 1.15]})
    vmin = max(0.5, np.floor(np.nanmin(M.values) * 20) / 20)
    im = a.imshow(M.values, cmap="viridis", vmin=vmin, vmax=1.0, aspect="auto")
    for i, l in enumerate(labs):
        th = train_hospital(l)
        for j, h in enumerate(hosp):
            v = M.values[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v < (vmin + 1) / 2 else "#1F2A44")
            if th and h in th.split("+"):
                a.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#FDAF91", lw=2.2))
    a.set_xticks(range(len(hosp))); a.set_xticklabels(hosp, fontsize=9)
    a.set_yticks(range(len(labs))); a.set_yticklabels([pretty(l) for l in labs], fontsize=9)
    a.set_xlabel("Test hospital", fontsize=10); a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02); cb.set_label("AUROC vs reference label", fontsize=9); cb.outline.set_visible(False)
    off = np.linspace(-0.27, 0.27, len(zs_list))
    for i, (sup, tgt) in enumerate(rows):
        if i % 2 == 0:
            b.axhspan(i - .5, i + .5, color="#F7F8FA", zorder=0)
        for (z, mk), o in zip(zs_list, off):
            r = ext[(ext.zero_shot == z) & (ext.supervised == sup) & (ext.target_hospital == tgt)]
            if r.empty:
                continue
            r = r.iloc[0]; c = EXT_COLOR[z]
            b.plot([r.lo, r.hi], [i + o, i + o], color=c, lw=1.2, zorder=2)
            b.scatter(r["diff"], i + o, marker=mk, s=30, color=c, edgecolors="white", linewidths=0.6, zorder=3)
    b.axvline(0, ls="--", color=REF, lw=0.8)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([f"{pretty(s).replace(' (trained on ', ', trained at ').rstrip(')')}, tested at {t}" for s, t in rows], fontsize=8.5)
    b.set_ylim(len(rows) - .5, -.5)
    b.set_xlabel("AUROC difference at a hospital not used for training (zero-shot minus supervised), 95% CI", fontsize=10)
    _style(b); b.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    b.text(0.99, 1.01, "Favors zero-shot", transform=b.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)
    b.text(0.01, 1.01, "Favors supervised", transform=b.transAxes, ha="left", va="bottom", fontsize=8.5, color=MUTED)
    h = [mlines.Line2D([], [], marker=mk, ls="", color=EXT_COLOR[z], markeredgecolor="white", ms=7, label=pretty(z)) for z, mk in zs_list]
    b.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=9, frameon=False)
    for i, ax in enumerate((a, b)):
        ax.text(-0.55 if i == 0 else -0.62, 1.03, "ab"[i], transform=ax.transAxes, fontsize=14, fontweight="bold", color=LABEL)
    fig.tight_layout()
    _save(fig, out, stem)
    return fig


def fig_range(native: pd.DataFrame, rng: pd.DataFrame, out, stem="eFig_hospital_range"):
    """Common cross-hospital range (maximum minus minimum hospital AUROC) for every labeler, 95% CI;
    the prespecified definition shown alongside for single-hospital supervised models."""
    from .analyze import train_hospital
    labs = rng.sort_values("gap").labeler.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(labs) + 1.2))
    nat = native.set_index("labeler")
    for i, l in enumerate(labs):
        r = rng.set_index("labeler").loc[l]; c = FAMILY_COLOR[family(l)]
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=1.5); ax.scatter(r.gap, i, s=40, color=c, edgecolors="white", zorder=3)
        if train_hospital(l) is not None and l in nat.index:
            ax.scatter(nat.loc[l, "gap"], i, s=34, marker="s", facecolors="white", edgecolors=c, linewidths=1.2, zorder=3)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([pretty(l) for l in labs], fontsize=8.5); ax.set_ylim(len(labs) - .5, -.5)
    ax.set_xlabel("Maximum minus minimum hospital AUROC (95% CI)", fontsize=10); _style(ax)
    h = [mlines.Line2D([], [], marker="o", ls="", color=LABEL, label="Common range (all labelers)"),
         mlines.Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=LABEL, label="Prespecified gap, own minus other hospitals")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.4, -0.07), ncol=2, fontsize=8.5, frameon=False)
    _save(fig, out, stem)
