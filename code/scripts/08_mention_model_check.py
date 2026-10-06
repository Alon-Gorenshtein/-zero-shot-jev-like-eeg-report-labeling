"""Post hoc (second revision): model-based check of the rule-based seizure mention audit, from the zero-shot labelers'
existing findings answers (no new inference). Writes output/tables/mention_model_check.csv (Table S23)."""
import pandas as pd
from eegrl import analyze as A
from eegrl.config import INTER, OUT
from eegrl.text import mention_class

LABS = ["kev-0.8b", "kev-4b", "kev-9b", "laya-en", "laya-multi", "nli-deberta", "gemma4-e4b", "gemma4-12b", "qwen38-27b"]
C = pd.read_parquet(INTER / "cohort.parquet", columns=["report_id", "text", "split", "fr_seizure"])
C = C[C.split == "test"]
preds = {l: pd.read_parquet(OUT / "predictions" / f"{l}-findings__full.parquet", columns=["report_id", "p_seizure"])
         for l in LABS}
out = A.mention_model_check(C, "seizure", mention_class, preds)
out.to_csv(OUT / "tables" / "mention_model_check.csv", index=False)
print(out[out.reference == 1].pivot_table(index="labeler", columns="mention", values=["prop", "n"]).round(3).to_string())
