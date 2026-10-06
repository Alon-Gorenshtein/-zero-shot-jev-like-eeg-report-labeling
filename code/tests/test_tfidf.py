from eegrl.labelers import tfidf


def test_tfidf_learns_simple_signal():
    X = ["This is an abnormal EEG with spikes."] * 20 + ["This is a normal EEG."] * 20
    y = [1] * 20 + [0] * 20
    m = tfidf.fit(X, y, min_df=1)
    p = m.predict_proba(["abnormal EEG with spikes", "normal EEG"])[:, 1]
    assert p[0] > .5 > p[1]


def test_tfidf_findings_fits_one_model_per_finding():
    import pandas as pd
    X = ["lateralized periodic discharges seen"] * 15 + ["normal background only"] * 15
    y = pd.DataFrame({"fr_lpd": [1] * 15 + [0] * 15, "fr_bs": [0] * 30})
    ms = tfidf.fit_findings(X, y, ["lpd", "bs"], min_df=1)
    p = tfidf.predict_findings(ms, ["periodic discharges lateralized", "normal background"])
    assert p[0]["p_lpd"] > .5 > p[1]["p_lpd"]
    assert p[0]["p_bs"] == 0.0  # constant-negative finding -> probability 0
