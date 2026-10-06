import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from ..prompts import QUESTION, FINDING_QUESTION, finding_key


def make_predictor(url="http://127.0.0.1:8009", prompt="primary", workers="4", timeout="300", task="binary",
                   retries="6", backoff="30"):
    if task == "findings":
        qs = {finding_key(f)[2:]: q for f, q in FINDING_QUESTION.items()}
    else:
        qs = {"q": QUESTION[prompt]}

    def one(text):
        # retry transient server failures (e.g., a restarted server); after `retries` attempts record a missing value
        for attempt in range(int(retries)):
            try:
                r = requests.post(f"{url}/v1/systemone", json={"state": text, "questions": qs}, timeout=float(timeout))
                r.raise_for_status()
                break
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                if attempt == int(retries) - 1:
                    print(f"systemone: giving up on one report after {retries} attempts: {e}", file=sys.stderr, flush=True)
                    keys = [f"p_{k}" for k in qs] if task == "findings" else ["p_abnormal"]
                    return {**{k: math.nan for k in keys}, "failed": True}
                time.sleep(float(backoff))
        ans = r.json()["answers"]
        if task == "findings":
            return {f"p_{k}": float(ans[k]["noul"]) for k in qs}
        return {"p_abnormal": float(ans["q"]["probabilities"]["abnormal"])}

    def predict(texts):
        with ThreadPoolExecutor(int(workers)) as ex:
            return list(ex.map(one, texts))
    return predict
