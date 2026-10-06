"""Open LLM labeler via a vLLM OpenAI-compatible server: exact sequence log-probability of each answer word."""
from concurrent.futures import ThreadPoolExecutor
import requests
from .llm_mlx import _chat, _chat_msgs, _p_abnormal
from ..prompts import LLM_FINDING_SYSTEM, LLM_FINDING_USER, FINDING_TEXT, finding_key


def _cont_logprob(prompt_logprobs, ids, n_prefix):
    return float(sum(prompt_logprobs[i][str(ids[i])]["logprob"] for i in range(n_prefix, len(ids))))


def make_predictor(model, url="http://127.0.0.1:8000", prompt="primary", max_tokens="16384", workers="32",
                   tokenizer=None, timeout="600", task="binary"):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
    words = ("no", "yes") if task == "findings" else ("normal", "abnormal")
    cont = {k: tokenizer.encode(k, add_special_tokens=False) for k in words}
    cap = int(max_tokens)

    def _post(seq, k):
        r = requests.post(f"{url}/v1/completions", json={
            "model": model, "prompt": seq, "max_tokens": 1, "temperature": 0, "prompt_logprobs": k},
            timeout=float(timeout))
        r.raise_for_status()
        return r.json()["choices"][0]["prompt_logprobs"]

    def score(chat_text):
        """P(second word) vs first word ('abnormal' vs 'normal', or 'yes' vs 'no') after the chat prompt."""
        ids = tokenizer.encode(chat_text, add_special_tokens=False)
        # One pass on prefix + the longer answer; the other (single-token) answer is read from the top-k
        # prompt logprobs at the first answer position. Fallback: a second pass when it is not in the top-k.
        longer, other = sorted(cont, key=lambda k: -len(cont[k]))
        seq = ids + cont[longer]
        plp = _post(seq, 20)
        lps = {longer: _cont_logprob(plp, seq, len(ids))}
        at = plp[len(ids)]
        if len(cont[other]) == 1 and str(cont[other][0]) in at:
            lps[other] = float(at[str(cont[other][0])]["logprob"])
        else:
            seq2 = ids + cont[other]
            lps[other] = _cont_logprob(_post(seq2, 0), seq2, len(ids))
        return _p_abnormal(lps[words[1]], lps[words[0]])

    def one(text):
        n = len(tokenizer.encode(_chat(tokenizer, text, prompt), add_special_tokens=False))
        trunc = n > cap
        if trunc:
            text = text[: cap * 3]
        if task == "findings":
            return {**{finding_key(f): score(_chat_msgs(tokenizer, LLM_FINDING_SYSTEM,
                                                         LLM_FINDING_USER.format(report=text, finding=t)))
                       for f, t in FINDING_TEXT.items()}, "truncated": trunc}
        return {"p_abnormal": score(_chat(tokenizer, text, prompt)), "truncated": trunc}

    def predict(texts):
        with ThreadPoolExecutor(int(workers)) as ex:
            return list(ex.map(one, texts))
    return predict
