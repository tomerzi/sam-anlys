"""
Gymnasium single-agent view of the same simulation (curriculum stages 0-2).

This is a wrapper, not a second simulation: robot 0 is controlled by the policy
while robot 1 holds the standing pose. Because it drives the identical
PassingSim, the observation and action spaces match the PettingZoo env exactly,
so a checkpoint trained here loads straight into the two-robot stages.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from sim.passing_sim import PassingSim, N_JOINTS, OBS_DIM
from sim import rewards as R


class SingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Config | None = None, seed: int = 0):
        self.config = config or Config()
        self.sim = PassingSim(self.config, seed=seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        self.sim.reset(passer=0)
        self.state = R.new_pass_state(self.sim)
        return self.sim.observe(0).astype(np.float32), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        cfg = self.config
        prev = self.sim.prev_action.copy()
        # Robot 1 is a passive stand-in: a zero action holds the standing pose.
        act = np.stack([np.asarray(action, dtype=np.float64), np.zeros(N_JOINTS)])

        self.sim.step(act)
        reward_vec, breakdown = R.compute_rewards(self.sim, self.state, act, prev)
        if R.update_pass_state(self.sim, self.state):
            reward_vec[0] += cfg.w_pass

        terminated = (self.sim.fallen(0)
                      or (cfg.hand_contact_terminates and self.state.fouled[0]))
        if cfg.stage >= 2 and np.linalg.norm(self.sim.ball_pos()[:2]) > cfg.ball_bounds:
            terminated = True
        truncated = self.sim.step_count >= cfg.max_episode_steps

        info = {"reward_terms": breakdown[0], "passes": self.state.passes}
        return (self.sim.observe(0).astype(np.float32), float(reward_vec[0]),
                bool(terminated), bool(truncated), info)

    def close(self):
        pass
