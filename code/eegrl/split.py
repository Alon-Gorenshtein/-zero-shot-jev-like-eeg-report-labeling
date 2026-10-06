import numpy as np
import pandas as pd
from .config import SEED


def assign_split(df: pd.DataFrame, seed: int = SEED, frac=(0.7, 0.1, 0.2)) -> pd.Series:
    """Patient-level split, stratified by site. BDSP patient IDs are site-specific, so the key is (site, patient)."""
    rng = np.random.default_rng(seed)
    pat_split = {}
    for site in sorted(df.site.unique()):
        pats = np.array(sorted(df.loc[df.site == site, "patient_id"].unique()))
        rng.shuffle(pats)
        n = len(pats)
        n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
        for i, p in enumerate(pats):
            pat_split[(site, p)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return pd.Series([pat_split[(s, p)] for s, p in zip(df.site, df.patient_id)], index=df.index)
