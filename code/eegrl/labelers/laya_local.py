from ..prompts import QUESTION, FINDING_QUESTION, finding_key


def make_predictor(model="english", prompt="primary", max_len="512", device="mps", task="binary"):
    """Laya Router. model: english (ModernBERT-large, 512) | multilingual (mmBERT-base, up to 8192)."""
    from laya import Router
    router = Router(device=device)
    n = int(max_len)
    qs = {finding_key(f)[2:]: q for f, q in FINDING_QUESTION.items()} if task == "findings" else {"q": QUESTION[prompt]}

    def predict(texts):
        out = []
        for t in texts:
            r = router.predict(t, qs, model=model, max_len=n)
            used = r.get("usage", {}).get("input_tokens", 0)
            if task == "findings":
                row = {f"p_{k}": float(r["answers"][k]["noul"]) for k in qs}
            else:
                row = {"p_abnormal": float(r["answers"]["q"]["probabilities"]["abnormal"])}
            out.append({**row, "truncated": used >= n})
        return out
    return predict
