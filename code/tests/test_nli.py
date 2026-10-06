from eegrl.labelers import nli


def test_label_ids_case_insensitive():
    assert nli._label_ids({0: "ENTAILMENT", 1: "neutral", 2: "Contradiction"}) == (2, 0)


def test_device_prefers_cuda(monkeypatch):
    import types
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True),
                                 backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)))
    assert nli._device(fake) == "cuda"
    fake.cuda.is_available = lambda: False
    assert nli._device(fake) == "mps"
