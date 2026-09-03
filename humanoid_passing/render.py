"""
Offscreen rendering of a trained policy to mp4.

Rendering runs in TWO PROCESSES on purpose. Importing torch and initialising an
OSMesa/EGL GL context in the same interpreter segfaults inside torch's CUDA
bindings, so:

  phase 1  roll the policy out with torch, record qpos per frame  (no GL)
  phase 2  replay that trajectory through mujoco.Renderer         (no torch)

Phase 2 re-invokes this file with --replay. The split also means a trajectory
can be re-rendered at any resolution or camera angle without re-running the
policy.

Watching the video is not optional: every high-reward humanoid failure mode -
shuffling, diving at the ball, vibrating in place - looks perfectly healthy on a
TensorBoard curve.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

GL_HELP = """
  [Error] MuJoCo could not open an offscreen OpenGL context.

  Rendering needs a GL backend. Pick the one that matches your machine:

      export MUJOCO_GL=egl      # Linux with a GPU (Colab, Kaggle)
      export MUJOCO_GL=osmesa   # CPU-only Linux; needs libosmesa6
                                #   apt-get install -y libosmesa6 libgl1
      export MUJOCO_GL=glfw     # desktop Linux / macOS with a display

  Training and evaluation do not need any of this - only rendering does.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: roll the policy out and record the trajectory (torch, no GL)
# ══════════════════════════════════════════════════════════════════════════════
def rollout_qpos(config, seconds: float = 10.0, untrained: bool = False) -> np.ndarray:
    from config import Config
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
        else:
            print(f"  [Warn] no checkpoint at {ckpt}; recording the zero-action policy")

    single = config.stage <= 2
    env = SingleAgentEnv(config) if single else PassingParallelEnv(config)
    obs = env.reset(seed=0)[0]

    def act(observation):
        if model is None:
            return np.zeros(20)
        source = normalizer(observation) if normalizer else observation
        return model.predict(source, deterministic=True)[0]

    trajectory = [env.sim.data.qpos.copy()]
    for _ in range(int(seconds / config.control_dt)):
        if single:
            obs, _, term, trunc, _ = env.step(act(obs))
            done = term or trunc
        else:
            obs, _, term, trunc, _ = env.step({a: act(o) for a, o in obs.items()})
            done = all(term.values()) or all(trunc.values())
        trajectory.append(env.sim.data.qpos.copy())
        if done:
            break

    print(f"[Render] rolled out {len(trajectory)} frames")
    return np.array(trajectory)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: replay a trajectory through the renderer (GL, no torch)
# ══════════════════════════════════════════════════════════════════════════════
def frames_from_qpos(scene_xml: Path, qpos: np.ndarray, width: int = 640,
                     height: int = 480, stride: int = 1) -> list[np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:                      # noqa: BLE001 - surface a usable message
        raise RuntimeError(f"{GL_HELP}\n  Underlying error: {exc}") from exc

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.elevation, camera.azimuth = 2.2, -12.0, 90.0
    camera.lookat[:] = [0.0, 0.0, 0.25]

    frames = []
    selected = qpos[::stride]
    for index, q in enumerate(selected):
        if index % 25 == 0:
            print(f"  [Render] frame {index}/{len(selected)}", flush=True)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        frames.append(renderer.render())
    renderer.close()
    return frames


# ------------------------------------------------------------------
def render(config, out: Path, seconds: float = 10.0, untrained: bool = False) -> Path:
    """Roll out in this process, then render in a clean subprocess."""
    qpos = rollout_qpos(config, seconds=seconds, untrained=untrained)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
        traj_path = Path(handle.name)
    np.save(traj_path, qpos)

    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--replay", str(traj_path), "--out", str(out),
           "--scene", str(config.scene_xml), "--fps", str(int(1.0 / config.control_dt))]
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    result = subprocess.run(cmd, env=env, cwd=str(Path(__file__).parent))
    traj_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"rendering subprocess failed (exit {result.returncode}){GL_HELP}")
    return out


# ------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Render a trajectory to mp4")
    parser.add_argument("--replay", type=Path, required=True, help="qpos .npy from phase 1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    # OSMesa software rendering of the OP3's high-poly visual meshes is slow;
    # stride>1 keeps the motion readable at a fraction of the cost.
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    import imageio.v2 as imageio
    frames = frames_from_qpos(args.scene, np.load(args.replay),
                              args.width, args.height, args.stride)
    imageio.mimsave(args.out, frames, fps=max(1, args.fps // args.stride))
    print(f"[Render] {len(frames)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
