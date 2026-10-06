from ..text import verdict


def predict(texts):
    return [verdict(t) for t in texts]


def make_predictor(**_):
    return lambda texts: [{"p_abnormal": p} for p in predict(texts)]
