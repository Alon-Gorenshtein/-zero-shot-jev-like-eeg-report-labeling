import re
import zipfile
from collections import defaultdict
import numpy as np
import pandas as pd
from .config import LABEL_CSVS, RAW, FINDING_COLS

NAME = "DeidentifiedName(Reports)"
_YEAR = re.compile(r"_(\d{4})\d{4}\.txt$")


def _flag(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def _col(d: pd.DataFrame, c: str) -> pd.Series:
    return d[c] if c in d else pd.Series(index=d.index, dtype=object)


def load_label_table(csvs: dict = LABEL_CSVS) -> pd.DataFrame:
    frames = []
    for (site, sub), path in csvs.items():
        d = pd.read_csv(path, low_memory=False, dtype=str)
        d = d[d[NAME].notna()].copy()
        d["site"], d["subsite"] = site, sub
        d["_n"], d["_a"] = _flag(_col(d, "normal")), _flag(_col(d, "abnormal"))
        for c in FINDING_COLS:
            v = _col(d, c).fillna("").astype(str)
            d[f"f_{c}"] = _flag(_col(d, c))
            d[f"fr_{c}"] = v.str.contains("report")
            d[f"fa_{c}"] = v.str.contains("annotation|verified")
            d[f"fv_{c}"] = v.str.contains("verified")  # HEEDB highest-reliability tier (positives only)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby(NAME)
    n_any, a_any = g["_n"].any(), g["_a"].any()
    conflict = n_any & a_any
    if conflict.any():
        raise ValueError(f"conflicting normal/abnormal labels for {int(conflict.sum())} reports, "
                         f"e.g. {conflict[conflict].index[0]}")
    first = g.first()
    out = pd.DataFrame({
        "report_id": first.index,
        "site": first["site"].values,
        "subsite": first["subsite"].values,
        "patient_id": first["BDSPPatientID"].astype(str).values,
        "label": np.where(a_any.values, 1, np.where(n_any.values, 0, -1)),
        "age": pd.to_numeric(first["AgeAtVisit"], errors="coerce").values,
        "sex": _col(first, "SexDSC").astype(str).str[0].str.upper().values,
        "service": _col(first, "ServiceName(EEG)").values,
    })
    days = pd.to_numeric(_col(first, "AgeInDaysAtVisit"), errors="coerce").values
    out["age_days"] = np.where(np.isnan(days), out["age"].values * 365.25, days)
    for c in FINDING_COLS:
        for pre in ("f_", "fr_", "fa_", "fv_"):
            out[f"{pre}{c}"] = g[f"{pre}{c}"].any().astype(int).values
    out = out[out.label >= 0].reset_index(drop=True)
    out["year"] = out.report_id.str.extract(_YEAR, expand=False).astype(float).astype("Int64")
    return out


def _is_report(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return "__MACOSX" not in name and not base.startswith("._") and base.endswith(".txt")


def build_zip_index(zips=None) -> dict:
    if zips is None:
        zips = sorted(p for p in (RAW / "reports").rglob("*.zip") if not p.name.startswith("._"))
    idx = {}
    for z in zips:
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                if _is_report(n):
                    idx.setdefault(n.rsplit("/", 1)[-1], str(z))
    return idx


def read_reports(ids, index: dict) -> dict:
    by_zip = defaultdict(set)
    for i in ids:
        by_zip[index[i]].add(i)
    out = {}
    for z, want in by_zip.items():
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                base = n.rsplit("/", 1)[-1]
                if base in want and base not in out and _is_report(n):
                    raw = f.read(n)
                    out[base] = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return out
