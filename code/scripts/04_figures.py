import json
import pandas as pd
from eegrl import figures as F, analyze as A
from eegrl.config import OUT

T, FIG = OUT / "tables", OUT / "figures"
prim = pd.read_csv(T / "primary_agreement.csv")
xf = pd.read_csv(T / "transfer_matrix.csv"); gp = pd.read_csv(T / "generalization_gap.csv")
main = lambda d: d[~d.labeler.str.contains("-512|-para")]
F.fig_transfer(main(xf), main(gp), FIG)  # prespecified-gap version (superseded; kept for the record)
if (T / "external_paired.csv").exists():  # written by 07_revision.py
    ext = pd.read_csv(T / "external_paired.csv")
    F.fig_external(main(xf), ext, FIG, principal=["kev-4b__full", "kev-9b__full", "nli-deberta__full"], comparators=("tfidf",))
    F.fig_external(main(xf), ext, FIG, stem="eFig_external_full")
    F.fig_range(main(gp), main(pd.read_csv(T / "generalization_gap_range.csv")), FIG)
F.fig_forest(main(prim), FIG)
F.fig_masked(main(prim), FIG, subset=pd.read_csv(T / "primary_agreement_subset6000.csv"))
sub = pd.read_csv(T / "subgroups.csv")
top = prim[prim.labeler.str.endswith("__full") & ~prim.labeler.str.contains("-para|tfidf-train")].copy()
top["fam"] = top.labeler.map(F.family)
keep = top.sort_values("auroc", ascending=False).groupby("fam").head(2).labeler.tolist()
F.fig_heatmap(sub[sub.labeler.isin(keep)], FIG)
W = A.load_predictions("test")
cal = [l for l in keep if not l.startswith("rule")]
F.fig_calibration(W, cal, FIG)
st = json.load(open(OUT / "stats_digest.json"))
F.fig_latent(st["latent_class_all"], FIG)
F.fig_cost(main(prim), pd.read_csv(T / "speed_benchmark_spark.csv"), FIG)
print("figures written to", FIG)
