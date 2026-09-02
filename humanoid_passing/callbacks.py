"""
Training callbacks.

RewardTermLogger is the most useful debugging tool in this project. A humanoid
reward almost always fails by one term silently dominating the rest - the policy
then optimises that term and ignores the task, while the aggregate return curve
looks perfectly healthy. Logging each term separately makes that visible.
"""
from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardTermLogger(BaseCallback):
    """Writes the mean of every individual reward term to TensorBoard."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._sums: dict[str, float] = defaultdict(float)
        self._count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            terms = info.get("reward_terms")
            if not terms:
                continue
            for name, value in terms.items():
                self._sums[name] += float(value)
            if "passes" in info:
                self._sums["passes"] += float(info["passes"])
            self._count += 1

        if self._count and self.n_calls % self.log_freq == 0:
            for name, total in self._sums.items():
                self.logger.record(f"reward/{name}", total / self._count)
            self._sums.clear()
            self._count = 0
        return True


class StageSuccessLogger(BaseCallback):
    """Tracks episode length and completed passes, the curriculum's success metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ep_lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_lengths.append(int(ep["l"]))
        if self.ep_lengths and self.n_calls % 2000 == 0:
            self.logger.record("stage/mean_ep_len", float(np.mean(self.ep_lengths[-100:])))
        return True
