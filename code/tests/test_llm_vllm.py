import math
from eegrl.labelers import llm_vllm


def test_cont_logprob_sums_continuation_positions():
    ids = [5, 6, 7, 8, 9]            # prefix = 5,6,7 ; continuation = 8,9
    plp = [None, {"6": {"logprob": -1.0}}, {"7": {"logprob": -1.0}},
           {"8": {"logprob": -0.5}, "3": {"logprob": -2.0}}, {"9": {"logprob": -0.25}}]
    assert abs(llm_vllm._cont_logprob(plp, ids, 3) - (-0.75)) < 1e-12


class Resp:
    def __init__(self, j): self._j = j
    def raise_for_status(self): pass
    def json(self): return self._j


def test_predict_posts_two_prompts_and_returns_probability(monkeypatch):
    sent = []

    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw): return "P"
        def encode(self, s, add_special_tokens=False): return {"P": [1, 2], "normal": [7], "abnormal": [6, 7]}[s]

    def fake_post(url, json, timeout):
        sent.append(json)
        # prompt = prefix(1,2) + abnormal(6,7); position 2 carries top-k incl. "normal" (7)
        plp = [None, {"2": {"logprob": 0.0}}, {"6": {"logprob": math.log(.6)}, "7": {"logprob": math.log(.2)}},
               {"7": {"logprob": 0.0}}]
        return Resp({"choices": [{"index": 0, "prompt_logprobs": plp}]})
    monkeypatch.setattr(llm_vllm.requests, "post", fake_post)
    fn = llm_vllm.make_predictor(model="m", url="http://x", tokenizer=Tok(), workers="1")
    out = fn(["report"])
    assert len(sent) == 1 and sent[0]["prompt"] == [1, 2, 6, 7] and sent[0]["prompt_logprobs"] == 20
    assert abs(out[0]["p_abnormal"] - .75) < 1e-9 and out[0]["truncated"] is False


def test_falls_back_to_second_request_when_normal_not_in_topk(monkeypatch):
    sent = []

    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw): return "P"
        def encode(self, s, add_special_tokens=False): return {"P": [1, 2], "normal": [7], "abnormal": [6, 7]}[s]

    def fake_post(url, json, timeout):
        sent.append(json)
        if json["prompt"] == [1, 2, 6, 7]:
            plp = [None, {"2": {"logprob": 0.0}}, {"6": {"logprob": math.log(.6)}}, {"7": {"logprob": 0.0}}]
        else:
            plp = [None, {"2": {"logprob": 0.0}}, {"7": {"logprob": math.log(.2)}}]
        return Resp({"choices": [{"index": 0, "prompt_logprobs": plp}]})
    monkeypatch.setattr(llm_vllm.requests, "post", fake_post)
    out = llm_vllm.make_predictor(model="m", url="http://x", tokenizer=Tok(), workers="1")(["r"])
    assert len(sent) == 2 and abs(out[0]["p_abnormal"] - .75) < 1e-9


def test_findings_mode_scores_yes_per_finding(monkeypatch):
    sent = []

    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw): return msgs[-1]["content"]
        def encode(self, s, add_special_tokens=False):
            return {"yes": [9], "no": [8]}.get(s, [1, 2])

    def fake_post(url, json, timeout):
        sent.append(json)
        plp = [None, {"2": {"logprob": 0.0}}, {"9": {"logprob": math.log(.3)}, "8": {"logprob": math.log(.1)}}]
        return Resp({"choices": [{"index": 0, "prompt_logprobs": plp}]})
    monkeypatch.setattr(llm_vllm.requests, "post", fake_post)
    out = llm_vllm.make_predictor(model="m", url="http://x", tokenizer=Tok(), workers="1", task="findings")(["r"])
    assert len(sent) == 10
    assert abs(out[0]["p_lpd"] - .75) < 1e-9 and "p_abnormal" not in out[0]
