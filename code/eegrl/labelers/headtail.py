"""Head+tail truncation for 512-token encoders (keep the first `head` tokens and the last rest), so the model
sees both the report opening and its closing impression. Sun et al., 2019 (How to fine-tune BERT for text
classification) found head+tail the best single-window strategy for long documents."""
import numpy as np


def head_tail(ids, n, head=128):
    if len(ids) <= n:
        return list(ids)
    return list(ids[:head]) + list(ids[len(ids) - (n - head):])


def _with_specials(tok, a, b=None):
    """[CLS] a [SEP] (b [SEP]) — the input format of BERT and DeBERTa-v3."""
    return [tok.cls_token_id] + list(a) + [tok.sep_token_id] + ([] if b is None else list(b) + [tok.sep_token_id])


def encode(tok, texts, max_len=512, pair=None, head=128):
    body = tok(list(texts), add_special_tokens=False)["input_ids"]
    p = tok([pair], add_special_tokens=False)["input_ids"][0] if pair is not None else None
    budget = max_len - len(_with_specials(tok, [], p))  # specials + pair tokens
    rows = [_with_specials(tok, head_tail(b, budget, head), p) for b in body]
    ids = np.full((len(rows), max(len(r) for r in rows)), tok.pad_token_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = r; mask[i, :len(r)] = 1
    return {"input_ids": ids, "attention_mask": mask, "truncated": [len(b) > budget for b in body]}
