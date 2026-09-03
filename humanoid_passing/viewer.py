"""
Interactive MuJoCo viewer.

This needs a real display and therefore only runs on your own machine - never in
a headless container or a Colab/Kaggle notebook. Use `main.py render` for an mp4
when there is no screen.

    python main.py viewer --stage 0              # watch the trained policy
    python main.py viewer --stage 0 --untrained  # watch the zero-action stance

Drag to orbit, scroll to zoom, and press Space to pause.
"""
import os
import platform
import sys
import time

import numpy as np

from config import Config
from sim.passing_sim import N_JOINTS

VIEWER_HELP = """
  [Error] Could not open an interactive viewer window.

  mujoco.viewer needs a real display, so it cannot run over SSH without X
  forwarding, in a container, or in a notebook. On a machine with a screen it
  should just work. Where there is no display, render a video instead:

      python main.py render --stage 0 --out runs/stage0.mp4
"""


# ------------------------------------------------------------------
def _has_display() -> bool:
    """MuJoCo's GLFW backend aborts the process rather than raising when there is
    no display, so check before opening the window instead of catching after."""
    if platform.system() != "Linux":
        return True          # macOS and Windows always have a window server
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def launch(config: Config, untrained: bool = False) -> int:
    if not _has_display():
        print(VIEWER_HELP)
        print("  No DISPLAY/WAYLAND_DISPLAY is set, so this is a headless machine.")
        return 1

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        print(f"{VIEWER_HELP}\n  Underlying error: {exc}")
        return 1

    from envs.single_env import SingleAgentEnv
    from envs.parallel_env import PassingParallelEnv

    model = normalizer = None
    if not untrained:
        ckpt = config.stage_dir() / "policy.zip"
        stats = config.stage_dir() / "vecnormalize.pkl"
        if ckpt.exists():
            from stable_baselines3 import PPO
            from obs_norm import load_obs_normalizer
            model = PPO.load(ckpt, device=config.device)
            if stats.exists():
                normalizer = load_obs_normalizer(stats)
            print(f"[Viewer] policy loaded from {ckpt}")
        else:
            print(f"  [Warn] no checkpoint at {ckpt}; showing the zero-action stance")

    single = config.stage <= 2
    env = SingleAgentEnv(config) if single else PassingParallelEnv(config)
    obs = env.reset(seed=0)[0]

    def act(observation):
        if model is None:
            return np.zeros(N_JOINTS)
        source = normalizer(observation) if normalizer else observation
        return model.predict(source, deterministic=True)[0]

    print("[Viewer] drag to orbit, scroll to zoom, Space to pause, Esc to quit")
    try:
        handle = mujoco.viewer.launch_passive(env.sim.model, env.sim.data)
    except Exception as exc:                      # noqa: BLE001 - surface a usable message
        print(f"{VIEWER_HELP}\n  Underlying error: {exc}")
        return 1

    with handle as viewer:
        while viewer.is_running():
            start = time.time()
            if single:
                obs, _, term, trunc, _ = env.step(act(obs))
                done = term or trunc
            else:
                obs, _, term, trunc, _ = env.step({a: act(o) for a, o in obs.items()})
                done = all(term.values()) or all(trunc.values())

            viewer.sync()
            if done:
                print("[Viewer] episode ended - resetting")
                obs = env.reset()[0]

            # Play back at wall-clock speed rather than as fast as the CPU allows.
            remaining = config.control_dt - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)
    return 0


def main() -> int:
    return launch(Config())


if __name__ == "__main__":
    sys.exit(main())
