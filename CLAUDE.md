# CLAUDE.md — sam-anlys

Two independent projects live here. Do not entangle them.

| Package | What it is |
|---|---|
| `football_action_coach/` | CV pipeline: YOLO11 → ByteTrack → SAM 2 → ViTPose → XGBoost, grading football actions from video |
| `humanoid_passing/` | RL: two MuJoCo humanoids (Robotis OP3) learning to pass a ball, Gymnasium + PettingZoo |

## House conventions (both packages)

- **Config**: one `@dataclass Config` per package in `config.py`. No YAML, no argparse for
  hyperparameters. Instantiate it exactly once, in `main()`, and pass it down.
- **Imports**: flat and absolute *relative to the package directory*
  (`from config import Config`, `from sim.passing_sim import PassingSim`).
  **Everything is run from inside the package directory**, not the repo root.
- **Entrypoint**: `argparse` subparsers → thin dispatch → `main()` + `if __name__ == "__main__":`.
- **Logging**: bracketed `print()` tags — `[Detector]`, `[Train]`, `[Scene]`, `  [Warn]`.
  No `logging` module.
- **Type hints**: PEP 604/585 (`list[dict]`, `int | None`). Python ≥3.10.
- **Comments**: box-drawing section separators — `# ── Section ──────`.
- **Paths**: `pathlib.Path` everywhere.
- **Large artifacts are never committed** — models, data, checkpoints, meshes are
  downloaded on demand and git-ignored.

There is no formatter config, and the author aligns `=` in blocks. Do not run Black or
ruff-format over existing files; it would reformat unrelated lines.

## humanoid_passing — things worth knowing before editing

- **One scene, one sim.** `PassingSim` owns physics, observations and reward; the
  Gymnasium and PettingZoo envs are thin adapters over it. Reward logic lives in exactly
  one place — keep it that way.
- **Observation (84-D) and action (20-D) spaces must stay identical across all five
  curriculum stages.** That is what lets a stage-N checkpoint load into stage N+1. If you
  add an observation, add it to every stage.
- **Freeze joints by zeroing their action scale, never by shrinking the action vector.**
- **OP3 geoms are unnamed** (mesh-derived). Resolve groups via
  `contact.geom → model.geom_bodyid → body name`. `mj_name2id` on a geom will return -1.
- **The no-hands rule is a hard requirement**, not a tunable: ball contact with
  `{l,r}_sho_pitch_link`, `{l,r}_sho_roll_link`, `{l,r}_el_link` is a foul.
- **`python main.py check` before any training run.** It catches the bugs that otherwise
  cost hours of compute.
- **SB3 + MuJoCo is CPU-bound.** Do not suggest a GPU as a speedup for Path A.
- **Do not set `num_cpus>0` in `concat_vec_envs_v1`.** SuperSuit's multiprocessing
  deadlocks against MuJoCo (workers block on a pipe, CPUs idle). Single-threaded is both
  correct and faster here. `train.SeedShim` patches a second SuperSuit/SB3 gap.

## Verify changes with

```bash
cd humanoid_passing
python main.py check                              # 10 assertions, ~10 s
python main.py train --stage 0 --steps 20000 --n-envs 4   # ~1 min end-to-end
```
