"""
Offscreen rendering of a trained policy to mp4.

Works headless, so it runs on Colab and Kaggle. Watching the video is not
optional: every high-reward humanoid failure mode - shuffling, diving at the
ball, vibrating in place - looks perfectly healthy on a TensorBoard curve.
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from config import Config
from envs.single_env import SingleAgentEnv
from envs.parallel_env import PassingParallelEnv


GL_HELP = """
  [Error] MuJoCo could not open an offscreen OpenGL context.

  Rendering needs a GL backend. Pick the one that matches your machine:

      export MUJOCO_GL=egl      # Linux with a GPU (Colab, Kaggle)
      export MUJOCO_GL=osmesa   # CPU-only Linux; needs libosmesa6
                                #   apt-get install -y libosmesa6 libgl1
      export MUJOCO_GL=glfw     # desktop Linux / macOS with a display

  On Colab, also run:  !apt-get install -y libosmesa6 libgl1
  Training and evaluation do not need any of this - only rendering does.
"""


def _make_renderer(model, height: int, width: int):
    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:                      # noqa: BLE001 - surface a usable message
        raise RuntimeError(f"{GL_HELP}\n  Underlying error: {exc}") from exc


# ------------------------------------------------------------------
def rollout_frames(config: Config, model=None, normalizer=None,
                   seconds: float = 10.0, width: int = 640, height: int = 480):
    env = SingleAgentEnv(config) if config.stage <= 2 else PassingParallelEnv(config)
    renderer = _make_renderer(env.sim.model, height, width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.elevation, camera.azimuth = 2.5, -15.0, 90.0
    camera.lookat[:] = [0.0, 0.0, 0.25]

    single = config.stage <= 2
    obs = env.reset(seed=0)[0]
    frames = []
    for _ in range(int(seconds / config.control_dt)):
        if single:
            action = np.zeros(20) if model is None else model.predict(
                normalizer.normalize_obs(obs) if normalizer else obs, deterministic=True)[0]
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        else:
            acts = {}
            for a, o in obs.items():
                acts[a] = np.zeros(20) if model is None else model.predict(
                    normalizer.normalize_obs(o) if normalizer else o, deterministic=True)[0]
            obs, _, term, trunc, _ = env.step(acts)
            done = all(term.values()) or all(trunc.values())

        renderer.update_scene(env.sim.data, camera)
        frames.append(renderer.render())
        if done:
            break
    renderer.close()
    return frames


# ------------------------------------------------------------------
def render(config: Config, out: Path, seconds: float = 10.0, untrained: bool = False):
    model = normalizer = None
    if not untrained:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        ckpt = config.stage_dir() / "policy.zip"
        stats = config.stage_dir() / "vecnormalize.pkl"
        if ckpt.exists():
            model = PPO.load(ckpt, device=config.device)
            if stats.exists():
                normalizer = VecNormalize.load(
                    str(stats), DummyVecEnv([lambda: SingleAgentEnv(config)]))
                normalizer.training = False
        else:
            print(f"  [Warn] no checkpoint at {ckpt}; rendering the zero-action policy")

    frames = rollout_frames(config, model, normalizer, seconds=seconds)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=int(1.0 / config.control_dt))
    print(f"[Render] {len(frames)} frames -> {out}")


def main() -> int:
    config = Config()
    render(config, config.runs_dir / f"stage{config.stage}.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
