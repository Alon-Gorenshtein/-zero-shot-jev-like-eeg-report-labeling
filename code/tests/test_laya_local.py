import sys
import types
from eegrl.labelers import laya_local


def test_maps_model_and_reads_probability_and_truncation(monkeypatch):
    calls = []

    class FakeRouter:
        def __init__(self, device=None, **kw):
            self.device = device

        def predict(self, state, questions, model=None, max_len=None):
            calls.append((model, max_len, questions["q"]["type"]))
            return {"answers": {"q": {"probabilities": {"normal": .2, "abnormal": .8}}},
                    "usage": {"input_tokens": 700 if len(state) > 10 else 5}}
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Router=FakeRouter))
    fn = laya_local.make_predictor(model="multilingual", max_len="512", device="cpu")
    out = fn(["short", "a much longer report text"])
    assert out[0] == {"p_abnormal": .8, "truncated": False}
    assert out[1]["truncated"] is True
    assert calls[0] == ("multilingual", 512, "choice")


def test_laya_findings(monkeypatch):
    class FakeRouter:
        def __init__(self, device=None, **kw): pass

        def predict(self, state, questions, model=None, max_len=None):
            return {"answers": {k: {"noul": .7} for k in questions}, "usage": {"input_tokens": 10}}
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Router=FakeRouter))
    out = laya_local.make_predictor(task="findings", device="cpu")(["r"])
    assert out[0]["p_gen_slowing"] == .7 and len([k for k in out[0] if k.startswith("p_")]) == 10
