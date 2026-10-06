import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from ..config import INTER
from ..text import normalize
from ..config import TARGET_FINDINGS
from ..prompts import finding_key


def fit(texts, y, C=1.0, min_df=3):
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=200_000, sublinear_tf=True),
                      MaxAbsScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    return m.fit(texts, y)


def fit_findings(texts, Y, findings, C=1.0, min_df=3):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=200_000, sublinear_tf=True)
    X = MaxAbsScaler().fit(vec.fit_transform(texts))
    Xs = X.transform(vec.transform(texts))
    models = {}
    for f in findings:
        y = Y[f"fr_{f}"].values
        models[f] = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xs, y) if y.min() != y.max() else None
    return {"vec": vec, "scale": X, "models": models}


def predict_findings(ms, texts):
    Xs = ms["scale"].transform(ms["vec"].transform(texts))
    cols = {f: (m.predict_proba(Xs)[:, 1] if m is not None else [0.0] * len(texts)) for f, m in ms["models"].items()}
    return [{finding_key(f): float(cols[f][i]) for f in cols} for i in range(len(texts))]


def make_predictor(train_sites="all", model_path=None, task="binary"):
    if task == "findings":
        path = model_path or INTER / f"tfidf_findings_{train_sites}.joblib"
        try:
            ms = joblib.load(path)
        except FileNotFoundError:
            d = pd.read_parquet(INTER / "cohort.parquet")
            d = d[d.split == "train"]
            if train_sites != "all":
                d = d[d.hospital.isin(train_sites.split("+"))]
            ms = fit_findings([normalize(t) for t in d.text], d, TARGET_FINDINGS)
            joblib.dump(ms, path)
        return lambda texts: predict_findings(ms, texts)
    return _make_binary(train_sites, model_path)


def _make_binary(train_sites="all", model_path=None):
    path = model_path or INTER / f"tfidf_{train_sites}.joblib"
    try:
        m = joblib.load(path)
    except FileNotFoundError:
        d = pd.read_parquet(INTER / "cohort.parquet", columns=["text", "label", "split", "hospital"])
        d = d[d.split == "train"]
        if train_sites != "all":
            d = d[d.hospital.isin(train_sites.split("+"))]
        m = fit([normalize(t) for t in d.text], d.label.values)
        joblib.dump(m, path)
    return lambda texts: [{"p_abnormal": float(p)} for p in m.predict_proba(texts)[:, 1]]
