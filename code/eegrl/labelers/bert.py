import numpy as np
import pandas as pd
from ..config import INTER, SEED
from ..text import normalize
from .headtail import encode
from ..config import TARGET_FINDINGS
from ..prompts import finding_key

BASE = "emilyalsentzer/Bio_ClinicalBERT"


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _tensors(tok, texts, dev, max_len):
    import torch
    e = encode(tok, texts, max_len=max_len)
    return {k: torch.from_numpy(e[k]).to(dev) for k in ("input_ids", "attention_mask")}, e["truncated"]


def _predict(m, tok, texts, dev, max_len, bs=64, with_trunc=False):
    import torch
    m.eval(); out, tr = [], []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc, t = _tensors(tok, texts[i:i + bs], dev, max_len)
            out += m(**enc).logits.float().softmax(-1)[:, 1].cpu().tolist(); tr += t
    return (out, tr) if with_trunc else out


def _frames(d, hospitals="all"):
    if hospitals != "all":
        d = d[d.hospital.isin(hospitals.split("+"))]
    return d[d.split == "train"], d[d.split == "val"]


def _targets(d, findings=TARGET_FINDINGS):
    return d[[f"fr_{f}" for f in findings]].values.astype("float32")


def train_findings(cohort=INTER / "cohort.parquet", out_dir=INTER / "bert_findings", epochs=2, lr=2e-5, bs=16,
                   max_len=512, seed=SEED, hospitals="all"):
    """Multi-label Bio_ClinicalBERT (one sigmoid per target finding); early stopping on mean val AUROC."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from ..metrics import auroc
    torch.manual_seed(seed)
    dev = _device()
    d = pd.read_parquet(cohort, columns=["text", "split", "hospital"] + [f"fr_{f}" for f in TARGET_FINDINGS])
    tr, va = _frames(d, hospitals)
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(TARGET_FINDINGS),
                                                           problem_type="multi_label_classification").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    va_txt, Yva = [normalize(t) for t in va.text], _targets(va)
    best = -1.0
    for ep in range(epochs):
        m.train(); idx = np.random.default_rng(seed + ep).permutation(len(tr))
        Ytr = _targets(tr)
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            enc, _ = _tensors(tok, [normalize(t) for t in tr.text.values[j]], dev, max_len)
            loss = m(**enc, labels=torch.tensor(Ytr[j], device=dev)).loss
            loss.backward(); opt.step(); opt.zero_grad()
        P = np.array(_predict_multi(m, tok, va_txt, dev, max_len))
        a = np.nanmean([auroc(Yva[:, k], P[:, k]) for k in range(P.shape[1])])
        print(f"epoch {ep} val mean AUROC {a:.4f}", flush=True)
        if a > best:
            best = a; m.save_pretrained(out_dir); tok.save_pretrained(out_dir)


def _predict_multi(m, tok, texts, dev, max_len, bs=64, with_trunc=False):
    import torch
    m.eval(); out, tr = [], []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc, t = _tensors(tok, texts[i:i + bs], dev, max_len)
            out += torch.sigmoid(m(**enc).logits.float()).cpu().tolist(); tr += t
    return (out, tr) if with_trunc else out


def train(cohort=INTER / "cohort.parquet", out_dir=INTER / "bert", epochs=2, lr=2e-5, bs=16, max_len=512, seed=SEED,
          hospitals="all"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from ..metrics import auroc
    torch.manual_seed(seed)
    dev = _device()
    d = pd.read_parquet(cohort, columns=["text", "label", "split", "hospital"])
    tr, va = _frames(d, hospitals)
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    va_txt = [normalize(t) for t in va.text]
    best = -1.0
    for ep in range(epochs):
        m.train(); idx = np.random.default_rng(seed + ep).permutation(len(tr))
        for i in range(0, len(idx), bs):
            b = tr.iloc[idx[i:i + bs]]
            enc, _ = _tensors(tok, [normalize(t) for t in b.text], dev, max_len)
            loss = m(**enc, labels=torch.tensor(b.label.values, device=dev)).loss
            loss.backward(); opt.step(); opt.zero_grad()
        a = auroc(va.label.values, np.array(_predict(m, tok, va_txt, dev, max_len)))
        print(f"epoch {ep} val AUROC {a:.4f}", flush=True)
        if a > best:
            best = a; m.save_pretrained(out_dir); tok.save_pretrained(out_dir)


def make_predictor(model_dir=str(INTER / "bert"), max_len="512", task="binary"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
    if task == "findings":
        def predict_f(texts):
            P, tr = _predict_multi(m, tok, texts, dev, int(max_len), with_trunc=True)
            return [{**{finding_key(f): p[k] for k, f in enumerate(TARGET_FINDINGS)}, "truncated": t} for p, t in zip(P, tr)]
        return predict_f
    def predict(texts):
        p, tr = _predict(m, tok, texts, dev, int(max_len), with_trunc=True)
        return [{"p_abnormal": a, "truncated": t} for a, t in zip(p, tr)]
    return predict
