import numpy as np
from eegrl import dawid_skene as DS


def test_recovers_simulated_sens_spec():
    rng = np.random.default_rng(0); n = 20000
    z = (rng.random(n) < .65).astype(int)
    sens = np.array([.95, .85, .75, .9]); spec = np.array([.9, .95, .8, .7])
    L = np.column_stack([np.where(z == 1, rng.random(n) < s, rng.random(n) > c) for s, c in zip(sens, spec)]).astype(float)
    L[rng.random(L.shape) < .05] = np.nan
    r = DS.fit(L)
    assert abs(r["prevalence"] - .65) < .02
    assert np.allclose(r["sens"], sens, atol=.02) and np.allclose(r["spec"], spec, atol=.02)
