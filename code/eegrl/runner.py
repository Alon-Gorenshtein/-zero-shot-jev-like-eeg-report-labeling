import time
from pathlib import Path
import pandas as pd
from .text import mask_verdict, impression_first, normalize


def prepare_texts(df: pd.DataFrame, variant: str) -> list[str]:
    if variant == "full":
        return [normalize(t) for t in df.text]
    if variant == "masked":
        return [mask_verdict(t)[0] for t in df.text]
    if variant == "impression_first":
        return [impression_first(t) for t in df.text]
    raise ValueError(variant)


def _parts(out: Path) -> list[Path]:
    return sorted(p for p in out.parent.glob(out.stem + ".part*.parquet") if not p.name.startswith("._"))


def _done(out: Path) -> set:
    ids = set()
    for p in _parts(out) + ([out] if out.exists() else []):
        ids |= set(pd.read_parquet(p, columns=["report_id"]).report_id)
    return ids


def run(labeler: str, predict_fn, df: pd.DataFrame, out, variant: str, batch: int = 64):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = df[~df.report_id.isin(_done(out))]
    k = len(_parts(out))
    for i in range(0, len(todo), batch):
        chunk = todo.iloc[i:i + batch]
        texts = prepare_texts(chunk, variant)
        t0 = time.perf_counter()
        preds = predict_fn(texts)
        dt = (time.perf_counter() - t0) / max(len(texts), 1)
        keys = sorted({k for p in preds for k in p if k.startswith("p_")})
        rows = pd.DataFrame({"report_id": chunk.report_id.values,
                             **{k: [float(p.get(k, float("nan"))) for p in preds] for k in keys},
                             "truncated": [bool(p.get("truncated", False)) for p in preds],
                             "seconds": dt, "variant": variant, "labeler": labeler})
        rows.to_parquet(out.parent / f"{out.stem}.part{k:06d}.parquet", index=False)
        k += 1
    parts = _parts(out)
    frames = [pd.read_parquet(p) for p in parts] + ([pd.read_parquet(out)] if out.exists() else [])
    if frames:
        pd.concat(frames).drop_duplicates("report_id").to_parquet(out, index=False)
        for p in parts:
            p.unlink()
