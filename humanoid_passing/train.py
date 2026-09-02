"""
Stage-aware PPO training.

Stages 0-2 train a single robot through the Gymnasium wrapper; stages 3-4 train
both robots through the PettingZoo env with parameter sharing (one policy drives
both agents). Every stage warm-starts from the previous stage's checkpoint,
which is only possible because the observation and action spaces are identical
across all of them.
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecNormalize, VecEnvWrapper,
)

from config import Config
from callbacks import RewardTermLogger, StageSuccessLogger
from envs.single_env import SingleAgentEnv
from envs.parallel_env import PassingParallelEnv


# ══════════════════════════════════════════════════════════════════════════════
class SeedShim(VecEnvWrapper):
    """
    Compatibility shim between SuperSuit and stable-baselines3.

    SB3 >=2.x calls VecEnv.seed() during setup, but SuperSuit's ConcatVecEnv does
    not implement it, so PPO construction dies with an AttributeError. The env is
    already seeded at construction, so this just satisfies the interface.
    """

    def seed(self, seed: int | None = None):
        return [seed]

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


# ------------------------------------------------------------------
def _make_single(config: Config, rank: int):
    def _init():
        return Monitor(SingleAgentEnv(config, seed=config.seed + rank))
    return _init


def build_vec_env(config: Config):
    """Single-agent vec env for stages 0-2, parameter-shared multi-agent for 3-4."""
    if config.stage <= 2:
        cls = SubprocVecEnv if config.n_envs > 1 else DummyVecEnv
        return cls([_make_single(config, i) for i in range(config.n_envs)])

    import supersuit as ss
    env = PassingParallelEnv(config, seed=config.seed)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # Both agents are folded into one batch and driven by a single shared policy.
    # num_cpus=0 keeps SuperSuit single-threaded. Its multiprocessing path
    # deadlocks against MuJoCo here (workers block on a pipe, CPUs sit idle),
    # and MuJoCo stepping already releases the GIL, so this costs little.
    vec = ss.concat_vec_envs_v1(
        env, config.n_envs, num_cpus=0, base_class="stable_baselines3",
    )
    return SeedShim(vec)


# ------------------------------------------------------------------
def train(config: Config, resume: Path | None = None) -> Path:
    out = config.stage_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n[Train] stage {config.stage}  ->  {out}\n{'='*60}")

    venv = build_vec_env(config)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if resume is not None and Path(resume).exists():
        print(f"[Train] warm-starting from {resume}")
        model = PPO.load(resume, env=venv, device=config.device)
        # The reward changes shape between stages; stale Adam moments fight the
        # new objective, so drop the learning rate for the transfer.
        model.learning_rate = config.learning_rate * 0.5
        stats = Path(resume).with_name("vecnormalize.pkl")
        if stats.exists():
            venv = VecNormalize.load(str(stats), venv.venv)
            venv.training = True
            model.set_env(venv)
    else:
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            policy_kwargs={"net_arch": list(config.net_arch),
                           "log_std_init": config.log_std_init},
            tensorboard_log=str(config.runs_dir / "tb"),
            device=config.device,
            seed=config.seed,
            verbose=1,
        )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[RewardTermLogger(), StageSuccessLogger()],
        reset_num_timesteps=True,
        tb_log_name=f"stage{config.stage}",
    )

    ckpt = out / "policy.zip"
    model.save(ckpt)
    # Saving the normalisation statistics is not optional: reloading a policy
    # without them silently produces garbage.
    venv.save(str(out / "vecnormalize.pkl"))
    venv.close()
    print(f"[Train] saved {ckpt} and {out/'vecnormalize.pkl'}")
    return ckpt


# ------------------------------------------------------------------
def main() -> int:
    train(Config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
