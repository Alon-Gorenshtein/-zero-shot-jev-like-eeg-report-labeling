from eegrl.labelers import systemone


class Resp:
    def __init__(self, j): self._j = j
    def raise_for_status(self): pass
    def json(self): return self._j


def test_parses_choice_probabilities(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json)
        return Resp({"answers": {"q": {"type": "choice", "choice": "abnormal", "confidence": 0.9,
                                        "probabilities": {"normal": 0.1, "abnormal": 0.9}}}})
    monkeypatch.setattr(systemone.requests, "post", fake_post)
    fn = systemone.make_predictor(url="http://x", workers="1")
    out = fn(["This is an abnormal EEG."])
    assert out == [{"p_abnormal": 0.9}]
    assert sent[0]["state"] == "This is an abnormal EEG."
    assert sent[0]["questions"]["q"]["type"] == "choice"


def test_findings_task_sends_all_noul_questions(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json)
        return Resp({"answers": {k: {"type": "noul", "noul": 0.3} for k in json["questions"]}})
    monkeypatch.setattr(systemone.requests, "post", fake_post)
    out = systemone.make_predictor(url="http://x", workers="1", task="findings")(["r"])
    assert len(sent) == 1 and len(sent[0]["questions"]) == 10
    assert all(q["type"] == "noul" for q in sent[0]["questions"].values())
    assert out[0]["p_foc_slowing"] == 0.3 and out[0]["p_lpd"] == 0.3


def test_retries_then_nan_on_persistent_server_error(monkeypatch):
    import math
    calls = {"n": 0}

    class Bad:
        status_code = 500
        def raise_for_status(self): raise systemone.requests.HTTPError("500")
        def json(self): return {}

    def fake_post(url, json, timeout):
        calls["n"] += 1
        return Bad()
    monkeypatch.setattr(systemone.requests, "post", fake_post)
    monkeypatch.setattr(systemone.time, "sleep", lambda s: None)
    out = systemone.make_predictor(url="http://x", workers="1", retries="3")(["r"])
    assert calls["n"] == 3 and math.isnan(out[0]["p_abnormal"]) and out[0]["failed"] is True


def test_retry_succeeds_after_transient_error(monkeypatch):
    seq = iter([Exception("conn"), Resp({"answers": {"q": {"probabilities": {"normal": .4, "abnormal": .6}}}})])

    def fake_post(url, json, timeout):
        r = next(seq)
        if isinstance(r, Exception):
            raise systemone.requests.ConnectionError("down")
        return r
    monkeypatch.setattr(systemone.requests, "post", fake_post)
    monkeypatch.setattr(systemone.time, "sleep", lambda s: None)
    assert systemone.make_predictor(url="http://x", workers="1")(["r"])[0]["p_abnormal"] == .6
