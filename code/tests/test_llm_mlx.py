import math
from eegrl.labelers import llm_mlx


class TokNoThinking:
    def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
        return "|".join(m["content"] for m in msgs)


class TokThinking(TokNoThinking):
    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking=True):
        assert enable_thinking is False
        return "T:" + super().apply_chat_template(msgs, add_generation_prompt, tokenize)


def test_chat_disables_thinking_when_supported_and_falls_back():
    assert llm_mlx._chat(TokThinking(), "R", "primary").startswith("T:")
    s = llm_mlx._chat(TokNoThinking(), "RPT", "primary")
    assert "RPT" in s and "normal or abnormal" in s


def test_prob_from_logprobs():
    assert abs(llm_mlx._p_abnormal(math.log(.3), math.log(.1)) - .75) < 1e-12


def test_option_logprobs_one_pass_when_an_option_is_single_token():
    import numpy as np
    V = 5
    calls = []

    def forward(ids, start):  # fake log-softmax: row t puts .8 on token (ids[t]+1) % V
        calls.append(list(ids))
        rows = []
        for tok in ids:
            r = np.full(V, 0.05); r[(tok + 1) % V] = 0.8; rows.append(np.log(r))
        return np.array(rows)[start:]
    prefix = [0, 1, 2]
    conts = {"normal": [4], "abnormal": [3, 4]}
    lp = llm_mlx._option_logprobs(forward, prefix, conts)
    assert len(calls) == 1 and calls[0] == [0, 1, 2, 3, 4]
    # after token 2 the next token 3 has p=.8, token 4 has p=.05; after 3, token 4 has p=.8
    assert abs(lp["normal"] - np.log(.05)) < 1e-9
    assert abs(lp["abnormal"] - (np.log(.8) + np.log(.8))) < 1e-9


def test_option_logprobs_two_passes_when_both_multi_token():
    import numpy as np
    calls = []

    def forward(ids, start):
        calls.append(list(ids)); return np.log(np.full((len(ids), 6), 1 / 6))[start:]
    lp = llm_mlx._option_logprobs(forward, [0], {"normal": [1, 2], "abnormal": [3, 4]})
    assert len(calls) == 2 and abs(lp["normal"] - 2 * np.log(1 / 6)) < 1e-9
