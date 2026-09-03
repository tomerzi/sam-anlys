"""
Deterministic evaluation against each stage's success criterion.

Prints an explicit advance / do-not-advance verdict, so the curriculum is driven
by a measured result rather than by how the reward curve happens to look.
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config import Config
from obs_norm import load_obs_normalizer
from envs.single_env import SingleAgentEnv
from envs.parallel_env import PassingParallelEnv
from sim import rewards as R

# stage -> (metric name, threshold, human description)
CRITERIA = {
    0: ("survive_rate", 1.00, "every episode survives the full 20 s"),
    1: ("survive_rate", 0.90, ">90% survive 20 s while being pushed"),
    2: ("deliver_rate", 0.50, ">=50% deliver the ball to the target, upright"),
    3: ("pass_rate", 0.40, ">=40% complete a pass and trap"),
    4: ("mean_passes", 3.00, "mean >=3 consecutive passes per episode"),
}


# ------------------------------------------------------------------
def _load(config: Config, stage: int):
    out = config.stage_dir(stage)
    ckpt, stats = out / "policy.zip", out / "vecnormalize.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} - train stage {stage} first")
    if not stats.exists():
        raise FileNotFoundError(
            f"missing {stats}. A policy reloaded without its VecNormalize "
            f"statistics produces garbage, so evaluation refuses to run.")
    return PPO.load(ckpt, device=config.device), stats


# ------------------------------------------------------------------
def evaluate(config: Config, episodes: int = 20) -> dict[str, float]:
    model, stats = _load(config, config.stage)

    normalize = load_obs_normalizer(stats)
    env = SingleAgentEnv(config) if config.stage <= 2 else PassingParallelEnv(config)
    survived = delivered = passed = 0
    all_passes: list[int] = []

    for ep in range(episodes):
        if config.stage <= 2:
            obs, _ = env.reset(seed=1000 + ep)
        else:
            obs_d, _ = env.reset(seed=1000 + ep)
        done = False
        steps = 0
        while not done:
            if config.stage <= 2:
                action, _ = model.predict(normalize(obs), deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            else:
                acts = {}
                for a, o in obs_d.items():
                    action, _ = model.predict(normalize(o), deterministic=True)
                    acts[a] = action
                obs_d, _, term, trunc, info_d = env.step(acts)
                done = all(term.values()) or all(trunc.values())
                info = next(iter(info_d.values()))
            steps += 1

        full = steps >= config.max_episode_steps
        survived += int(full)
        passes = int(info.get("passes", 0))
        all_passes.append(passes)
        delivered += int(passes > 0)
        passed += int(passes > 0)

    metrics = {
        "survive_rate": survived / episodes,
        "deliver_rate": delivered / episodes,
        "pass_rate": passed / episodes,
        "mean_passes": float(np.mean(all_passes)),
    }

    key, threshold, description = CRITERIA[config.stage]
    value = metrics[key]
    print(f"\n{'='*60}")
    print(f"[Eval] stage {config.stage} over {episodes} episodes")
    for name, val in metrics.items():
        print(f"    {name:14s} {val:.3f}")
    print(f"  criterion: {description}")
    verdict = "PASS - advance to the next stage" if value >= threshold \
        else "FAIL - stay on this stage and fix it"
    print(f"  {key} = {value:.3f} vs {threshold:.2f}  ->  {verdict}")
    print(f"{'='*60}\n")
    return metrics


def main() -> int:
    evaluate(Config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
