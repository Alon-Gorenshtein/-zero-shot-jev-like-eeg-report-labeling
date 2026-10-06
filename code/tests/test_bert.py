import pandas as pd
from eegrl.labelers import bert


def test_training_frames_filter_by_hospital():
    d = pd.DataFrame({"text": list("abcdefgh"), "label": [0, 1] * 4,
                      "split": ["train", "train", "val", "val", "train", "val", "test", "train"],
                      "hospital": ["A", "A", "A", "B", "B", "B", "A", "A"]})
    tr, va = bert._frames(d, "A")
    assert list(tr.text) == ["a", "b", "h"] and list(va.text) == ["c"]
    tr, va = bert._frames(d, "all")
    assert len(tr) == 4 and len(va) == 3
    tr, va = bert._frames(d, "A+B")
    assert len(tr) == 4


def test_multilabel_targets():
    d = pd.DataFrame({"fr_spikes": [1, 0], "fr_lpd": [0, 1]})
    assert bert._targets(d, ["spikes", "lpd"]).tolist() == [[1.0, 0.0], [0.0, 1.0]]
