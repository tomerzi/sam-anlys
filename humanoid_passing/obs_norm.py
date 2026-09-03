"""
Loading VecNormalize statistics without building an environment.

VecNormalize.load() needs a live VecEnv purely to attach itself to, which costs
a full extra MuJoCo model (~1.5 GB with the OP3's meshes). Evaluation and
rendering only ever need the observation statistics, so they read the pickle
directly instead - three model copies in one process is enough to exhaust RAM
on a laptop and segfault.
"""
import pickle
from pathlib import Path
from typing import Callable

import numpy as np


# ------------------------------------------------------------------
def load_obs_normalizer(stats_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function applying the saved observation normalisation."""
    with open(stats_path, "rb") as handle:
        vec_normalize = pickle.load(handle)

    rms = vec_normalize.obs_rms
    mean, var = rms.mean, rms.var
    epsilon, clip = vec_normalize.epsilon, vec_normalize.clip_obs

    def normalize(obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - mean) / np.sqrt(var + epsilon), -clip, clip).astype(np.float32)

    return normalize
