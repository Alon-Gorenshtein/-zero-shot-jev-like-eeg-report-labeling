from ..prompts import NLI_HYPOTHESIS, NLI_FINDING_HYPOTHESIS, finding_key
from .headtail import encode


def _label_ids(id2label):
    inv = {v.lower(): int(k) for k, v in id2label.items()}
    return inv["contradiction"], inv["entailment"]


def _device(torch):
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_predictor(model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", prompt="primary", max_len="512",
                   task="binary"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = _device(torch)
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForSequenceClassification.from_pretrained(model).to(dev).eval()
    con, ent = _label_ids(m.config.id2label)
    h, n = NLI_HYPOTHESIS[prompt], int(max_len)

    hyps = {finding_key(f): hh for f, hh in NLI_FINDING_HYPOTHESIS.items()} if task == "findings" else {"p_abnormal": h}

    @torch.no_grad()
    def predict(texts):
        rows = [{} for _ in texts]
        for key, hyp in hyps.items():
            e = encode(tok, texts, max_len=n, pair=hyp)
            enc = {k: torch.from_numpy(e[k]).to(dev) for k in ("input_ids", "attention_mask")}
            p = m(**enc).logits[:, [con, ent]].float().softmax(-1)[:, 1].cpu().tolist()
            for r, pi, t in zip(rows, p, e["truncated"]):
                r[key] = pi; r["truncated"] = r.get("truncated", False) or t
        return rows
    return predict
