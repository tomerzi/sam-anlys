"""
Humanoid Ball Passing - two Robotis OP3 robots passing a ball with RL.

Usage
-----
    python main.py fetch                      # download the OP3 model from Menagerie
    python main.py build                      # generate the two-robot scene XML
    python main.py check                      # smoke test - run this before training
    python main.py train --stage 0            # train a curriculum stage
    python main.py train --stage 1 --resume   # warm-start from the previous stage
    python main.py eval  --stage 0
    python main.py render --stage 0 --out runs/stage0.mp4
    python main.py viewer --stage 0            # interactive window, local machine only

Everything is run from inside humanoid_passing/. The curriculum stages are
0 stand | 1 stand under push | 2 kick to target | 3 pass + receive | 4 rally.
"""
import argparse
import sys
from pathlib import Path

from config import Config


# ------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Humanoid Ball Passing (MuJoCo + RL)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="Download the Robotis OP3 model and meshes")
    sub.add_parser("build", help="Generate the two-robot passing scene")
    sub.add_parser("check", help="Run the smoke test")

    p = sub.add_parser("train", help="Train one curriculum stage")
    p.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="warm-start from stage-1")

    p = sub.add_parser("eval", help="Evaluate a stage against its success criterion")
    p.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=20)

    p = sub.add_parser("viewer", help="Interactive viewer (needs a display; local only)")
    p.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4])
    p.add_argument("--untrained", action="store_true")

    p = sub.add_parser("render", help="Render a policy to mp4")
    p.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--untrained", action="store_true", help="render the zero-action policy")

    args = parser.parse_args()
    config = Config()
    if getattr(args, "stage", None) is not None:
        config.stage = args.stage

    # ── Dispatch ──────────────────────────────────────────────────────
    if args.cmd == "fetch":
        from assets_fetch import fetch_op3
        return fetch_op3()

    if args.cmd == "build":
        from scene_builder import build_scene
        if not config.op3_xml.exists():
            print(f"  [Error] {config.op3_xml} missing - run: python main.py fetch")
            return 1
        build_scene(config)
        return 0

    if args.cmd == "check":
        from check import run_checks
        return 0 if run_checks(config) else 1

    if args.cmd == "train":
        from train import train
        if args.steps is not None:
            config.total_timesteps = args.steps
        if args.n_envs is not None:
            config.n_envs = args.n_envs
        if config.stage >= 3:
            config.arm_action_scale = config.action_scale   # arms unfrozen for balance
        resume = config.stage_dir(config.stage - 1) / "policy.zip" \
            if args.resume and config.stage > 0 else None
        train(config, resume=resume)
        return 0

    if args.cmd == "eval":
        from evaluate import evaluate
        evaluate(config, episodes=args.episodes)
        return 0

    if args.cmd == "viewer":
        from viewer import launch
        return launch(config, untrained=args.untrained)

    if args.cmd == "render":
        from render import render
        out = args.out or config.runs_dir / f"stage{config.stage}.mp4"
        render(config, out, seconds=args.seconds, untrained=args.untrained)
        return 0

    parser.error(f"unknown command {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
