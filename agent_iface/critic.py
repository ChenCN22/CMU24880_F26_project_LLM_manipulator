"""Early warning from the frozen skill's own critic.

The TQC critic estimates, at every control step, how much reward is still to
come.  With the sparse HER reward that is (minus) the number of steps the skill
expects to need.  A rollout that is going wrong shows a Q that stops rising.

The warning is a functional band fitted on SUCCESSFUL executions only:
  mu[t], sd[t]   mean and spread of -Q at step t over successful rollouts
  c              the 95th percentile, over those rollouts, of max_t z_t
A new rollout raises a warning at the first step whose z exceeds c.  By
construction about 5% of successful rollouts are flagged; how many failing ones
are flagged, and how early, is what the calibration script measures.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MIN_COUNT = 8          # steps with fewer successful rollouts reuse the last good band
SKIP = 3               # ignore the first steps, where Q reflects the start pose only


def q_value(policy, obs: Dict[str, np.ndarray], action: np.ndarray) -> float:
    import torch as th
    with th.no_grad():
        obs_t, _ = policy.policy.obs_to_tensor(obs)
        act_t = th.as_tensor(np.asarray(action, dtype=np.float32)[None], device=policy.device)
        q = policy.critic(obs_t, act_t)
        if isinstance(q, (tuple, list)):
            q = th.cat(list(q), dim=1)
        return float(q.mean())


def _z(trace: List[float], band: Dict[str, Any]) -> np.ndarray:
    mu, sd = np.asarray(band["mu"]), np.asarray(band["sd"])
    idx = np.minimum(np.arange(len(trace)), len(mu) - 1)
    z = (-np.asarray(trace, dtype=float) - mu[idx]) / sd[idx]
    z[:SKIP] = -np.inf
    return z


def fit(traces: List[List[float]], quantile: float = 0.95) -> Dict[str, Any]:
    """Fit the band on successful rollouts of one kind of call."""
    T = max(len(t) for t in traces)
    mu, sd = [], []
    for t in range(T):
        col = [-tr[t] for tr in traces if len(tr) > t]
        if len(col) < MIN_COUNT and mu:
            mu.append(mu[-1]); sd.append(sd[-1])
        else:
            mu.append(float(np.mean(col))); sd.append(float(max(np.std(col), 0.25)))
    band = {"mu": mu, "sd": sd, "n": len(traces), "quantile": quantile}
    band["c"] = float(np.quantile([_z(tr, band).max() for tr in traces], quantile))
    return band


def warning(calib: Optional[Dict[str, Any]], kind: str, trace: Optional[List[float]]
            ) -> Tuple[bool, Optional[int]]:
    """(fired, control step at which it fired)."""
    if not calib or kind not in calib or not trace or len(trace) <= SKIP:
        return False, None
    z = _z(trace, calib[kind])
    hits = np.nonzero(z > calib[kind]["c"])[0]
    return (True, int(hits[0])) if len(hits) else (False, None)


def load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)
