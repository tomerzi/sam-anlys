"""
PettingZoo ParallelEnv for the two-robot passing task (curriculum stages 3-4).

Both robots act simultaneously, which is the right API for continuous control;
the AEC/turn-based API would be a poor fit. The two agents are homogeneous and
share one policy during training (parameter sharing via SuperSuit), so their
observation and action spaces must be identical - which they are by construction.
"""
import functools

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from config import Config
from sim.passing_sim import PassingSim, N_JOINTS, OBS_DIM
from sim import rewards as R

AGENTS = ["robot_0", "robot_1"]


class PassingParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "op3_passing_v0"}

    def __init__(self, config: Config | None = None, seed: int = 0):
        self.config = config or Config()
        self.sim = PassingSim(self.config, seed=seed)
        self.possible_agents = list(AGENTS)
        self.agents = list(AGENTS)
        self._seed = seed
        self.render_mode = None      # SuperSuit's MarkovVectorEnv reads this

    # ------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        # A plain Box, never a Dict: SuperSuit's SB3 bridge rejects nested spaces.
        return spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        passer = int(self.sim.rng.integers(2)) if self.config.stage >= 3 else 0
        self.sim.reset(passer=passer)
        self.state = R.new_pass_state(self.sim)
        obs = {a: self._obs(i) for i, a in enumerate(self.agents)}
        return obs, {a: {} for a in self.agents}

    def _obs(self, i: int) -> np.ndarray:
        return self.sim.observe(i).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, actions: dict[str, np.ndarray]):
        cfg = self.config
        prev = self.sim.prev_action.copy()
        act = np.stack([np.asarray(actions[a], dtype=np.float64) for a in self.possible_agents])

        self.sim.step(act)
        reward_vec, breakdown = R.compute_rewards(self.sim, self.state, act, prev)
        completed = R.update_pass_state(self.sim, self.state)

        if completed:
            reward_vec += cfg.w_pass
            if cfg.stage >= 4:
                # Rally: the receiver becomes the passer and play continues.
                self.sim.passer = 1 - self.sim.passer
                self.state.prev_ball_dist = float(np.linalg.norm(
                    self.sim.ball_pos()[:2] - R.target_pos(self.sim, self.sim.passer)[:2]))
                self.state.last_toucher = -1
                self.state.delivered = False

        terminated = self._terminated()
        truncated = self.sim.step_count >= cfg.max_episode_steps
        # Both agents always end together: SuperSuit cannot handle one agent
        # disappearing from self.agents part-way through an episode.
        obs = {a: self._obs(i) for i, a in enumerate(self.possible_agents)}
        rew = {a: float(reward_vec[i]) for i, a in enumerate(self.possible_agents)}
        term = {a: terminated for a in self.possible_agents}
        trunc = {a: truncated for a in self.possible_agents}
        info = {a: {"reward_terms": breakdown[i], "passes": self.state.passes}
                for i, a in enumerate(self.possible_agents)}

        if terminated or truncated:
            self.agents = []
        return obs, rew, term, trunc, info

    def _terminated(self) -> bool:
        cfg = self.config
        if any(self.sim.fallen(i) for i in range(2)):
            return True
        if cfg.hand_contact_terminates and any(self.state.fouled):
            return True
        if cfg.stage >= 2:
            ball_xy = np.linalg.norm(self.sim.ball_pos()[:2])
            if ball_xy > cfg.ball_bounds:
                return True
        return False

    def render(self):
        raise NotImplementedError("Use render.py for offscreen video.")

    def close(self):
        pass
