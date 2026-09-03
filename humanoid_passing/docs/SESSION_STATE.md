# Session state — read this first

Handoff from the cloud session that built this package. `CLAUDE.md` covers the
conventions and invariants; this file covers **what has actually happened so far**, so you
do not re-derive settled decisions or re-introduce fixed bugs.

## Where the project stands

All code is complete and pushed on `claude/humanoid-ball-passing-rl-n1fejo`. All five
curriculum stages are implemented and both env adapters work. **No stage has been trained
to completion on real hardware.** Training is the next job.

## Measured, not assumed

These are real numbers from real runs. Trust them; do not re-measure.

| Fact | Value |
|---|---|
| OP3 zero-joint stance | torso settles at **0.279 m**, uprightness **1.000** |
| Throughput, single-agent | **~440 control steps/s**, `--n-envs 4` on 4 cores |
| Throughput, multi-agent | **~850 steps/s** (`num_cpus=0` beats multiprocessing) |
| Memory per env | **~1.5 GB** — the OP3 mesh assets dominate |
| Stage 0 learning curve | `ep_len_mean` 66 → 949 of 1000 by ~230k steps, monotonic |
| Stage 0 eval | `survive_rate 1.000` after only 20k steps |

The stable zero-pose is *why* actions are residuals around `NOMINAL_QPOS` (all zeros) and
why a zero action means "hold the stance". Do not change that without re-measuring.

`n_envs=8` (the config default) needs ~12 GB. On a laptop pass `--n-envs 4`.

**Not measured:** stages 1–4 on real hardware. Nothing past stage 0 has trained.

## Bugs already fixed — do not reintroduce

Each cost real debugging time. The symptom is listed so you recognise a regression.

- **`concat_vec_envs_v1(num_cpus>0)` deadlocks against MuJoCo.** Symptom: training hangs
  with load average near zero and CPU time barely advancing. Keep `num_cpus=0` — it is
  also ~2× faster here.
- **SuperSuit's `ConcatVecEnv` has no `.seed()`**, which SB3 ≥2.x calls during setup.
  `train.SeedShim` supplies it.
- **`MarkovVectorEnv` reads `render_mode`** off the raw env; `PassingParallelEnv` sets it
  to `None` for exactly this reason.
- **`PPO.load` segfaults when a GL context exists in the same process** (crash inside
  torch's CUDA bindings). This is why `render.py` splits rollout and rendering across two
  processes. Do not merge them back.
- **`VecNormalize.load` needs a live VecEnv** purely to attach to, costing another ~1.5 GB
  model; three copies in one process exhausted RAM. Use `obs_norm.load_obs_normalizer`.
- **`prev_ball_dist` must start at the true distance.** At 0.0 the first potential-shaping
  term equals `-dist`, handing every episode a large spurious penalty on step one.
  `check.py` asserts this.

## Next step

```bash
cd humanoid_passing
python main.py check                                  # expect 10/10 PASS
python main.py train --stage 0 --n-envs 4             # ~200k steps is enough
python main.py eval  --stage 0                        # prints an explicit verdict
python main.py viewer --stage 0                       # watch it (needs a display)
python main.py train --stage 1 --n-envs 4 --resume    # warm-start from stage 0
```

Advance a stage only when `eval` says PASS. When a stage stalls, open TensorBoard and look
at the `reward/*` per-term plots before touching any hyperparameter — one term dominating
the rest is the usual cause and is invisible in `ep_rew_mean`.

## Known gaps

- **Rendering is unverified on Windows.** It was only ever exercised with OSMesa on Linux.
  If `main.py render` misbehaves, the `MUJOCO_GL` notes in `README.md` are the first stop.
- **`viewer` has never run on a machine with a display.** It was only confirmed to exit
  cleanly when headless. It should work; it is not proven.
- **Stage 4 (continuous rally) is research-grade** and may not converge on CPU. Stages 0–3
  are the defensible deliverable; the MJX path in `README.md` is the escalation.
