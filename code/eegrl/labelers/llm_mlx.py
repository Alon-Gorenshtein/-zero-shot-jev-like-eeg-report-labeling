import math
from ..prompts import LLM_SYSTEM, LLM_USER


def _chat(tok, report, prompt):
    return _chat_msgs(tok, LLM_SYSTEM, LLM_USER[prompt].format(report=report))


def _chat_msgs(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = {"add_generation_prompt": True, "tokenize": False}
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)


def _p_abnormal(lp_abnormal, lp_normal):
    return 1 / (1 + math.exp(lp_normal - lp_abnormal))


def _option_logprobs(forward, prefix_ids, conts):
    """Sequence log-probability of each continuation. forward(ids, start) -> log-softmax rows start..len(ids)-1.
    One forward pass suffices when every other option is a single token (scored at the prefix's last row)."""
    n = len(prefix_ids)
    names = sorted(conts, key=lambda k: -len(conts[k]))
    out, done = {}, set()
    for k in names:
        if k in done:
            continue
        lp = forward(prefix_ids + conts[k], n - 1)
        out[k] = float(sum(lp[j, t] for j, t in enumerate(conts[k])))
        done.add(k)
        for o in names:
            if o not in done and len(conts[o]) == 1:
                out[o] = float(lp[0, conts[o][0]]); done.add(o)
    return out


def _mlx_forward(model):
    import mlx.core as mx
    import numpy as np

    def forward(ids, start):
        logits = model(mx.array([ids]))[0, start:].astype(mx.float32)
        return np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    return forward


def make_predictor(model="mlx-community/gemma-4-12B-it-4bit", prompt="primary", max_tokens="16384"):
    from mlx_lm import load
    m, tok = load(model)
    cont = {k: tok.encode(k, add_special_tokens=False) for k in ("normal", "abnormal")}
    cap = int(max_tokens)
    fwd = _mlx_forward(m)

    def predict(texts):
        out = []
        for t in texts:
            ids = tok.encode(_chat(tok, t, prompt), add_special_tokens=False)
            trunc = len(ids) > cap
            if trunc:
                ids = tok.encode(_chat(tok, t[: cap * 3], prompt), add_special_tokens=False)
            lp = _option_logprobs(fwd, ids, cont)
            out.append({"p_abnormal": _p_abnormal(lp["abnormal"], lp["normal"]), "truncated": trunc})
        return out
    return predict
