# Humanoid Ball Passing (MuJoCo + Gymnasium + PettingZoo)

Two [Robotis OP3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotis_op3)
humanoid robots learn to stay standing and pass a ball to each other, trained with
reinforcement learning through a staged curriculum.

**Soccer rules: a robot may never touch the ball with its hands or arms.** Feet, legs,
torso and head are legal. The rule is enforced in physics (contact detection), in the
reward, and by episode termination — not merely documented.

---

## Quickstart

### Linux / macOS

```bash
cd humanoid_passing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py fetch     # download the OP3 model (~45 MB, not vendored)
python main.py build     # generate the two-robot scene XML
python main.py check     # smoke test — always run this before training

python main.py train  --stage 0
python main.py eval   --stage 0
python main.py viewer --stage 0                      # interactive window (local machine)
python main.py render --stage 0 --out runs/stage0.mp4  # mp4 (works headless)
```

### Windows (PowerShell)

Work from a normal user folder — `C:\Windows\System32` is write-protected and
`git clone` fails there with "Permission denied".

**Recommended: let `uv` pin the Python version.** This sidesteps a broken or
missing system Python entirely — uv downloads its own interpreter for this
project and never touches your PATH:

```powershell
cd $HOME\Documents
git clone -b claude/humanoid-ball-passing-rl-n1fejo https://github.com/tomerzi/sam-anlys.git
cd sam-anlys\humanoid_passing

uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Pin **3.12**, not the newest release. `supersuit` is the least actively
maintained dependency here and is usually the first to lack wheels for a fresh
Python; without a wheel, pip falls back to building from source, which on
Windows is its own adventure.

Without uv, using the system Python:

```powershell
cd $HOME\Documents
git clone -b claude/humanoid-ball-passing-rl-n1fejo https://github.com/tomerzi/sam-anlys.git
cd sam-anlys\humanoid_passing

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python main.py fetch
python main.py build
python main.py check
python main.py viewer --stage 0 --untrained
```

There is no `source` on Windows and paths use `\`. Prefer `py` over `python` (the
launcher is more reliable) and always `python -m pip`, never a bare `pip`, so you
get the virtualenv's pip.

Two common Windows snags:

- **`Activate.ps1 cannot be loaded ...`** — PowerShell's execution policy. Run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.
- **`Failed to launch python.exe (0x80070003)`** — PATH points at a Python that no
  longer exists, usually a leftover Microsoft Store alias. Reinstall from
  python.org with *Add python.exe to PATH* ticked, and turn off the `python.exe`
  / `python3.exe` aliases under Settings → Apps → Advanced app settings →
  App execution aliases.

`viewer` opens a real MuJoCo window and needs a display, so it only works on your
own machine - not over SSH without X forwarding, not in a container, not in a
notebook. It says so and exits cleanly rather than crashing. Use `render`
anywhere headless.

Then walk up the curriculum, warm-starting each stage from the last:

```bash
python main.py train --stage 1 --resume
python main.py train --stage 2 --resume
python main.py train --stage 3 --resume
python main.py train --stage 4 --resume
```

---

## Read this before you rent a GPU

**A GPU does not speed up stable-baselines3 + MuJoCo.** The bottleneck is stepping
physics on the CPU; the policy is a small MLP and costs almost nothing. Moving SB3 to a
Colab or Kaggle GPU buys you close to nothing, and it is the most common way students
lose weeks on a project like this.

| Path | Stack | Throughput | Use for |
|---|---|---|---|
| **A (default)** | SB3 PPO + `SubprocVecEnv`, CPU | ~460 steps/s measured on 4 cores | stages 0–3 |
| **B (scale-up)** | MJX + Brax PPO, JAX on GPU | orders of magnitude faster | stage 4, if A plateaus |

This repo implements Path A. Use more CPU cores (`--n-envs`), not a bigger GPU.

**Measured on this codebase**, 4 envs on 4 cores: ~460 control steps/s (each is 10 physics
steps, so ~4.6k physics steps/s). The OP3's mesh collision geometry is what costs — it is
a detailed model, not a primitive-shape humanoid. Practical planning numbers:

| Budget | Wall clock at ~460 steps/s |
|---|---|
| 1 M steps | ~35 min |
| 3 M steps | ~1.8 h |
| 5 M steps | ~3 h |

Scale roughly linearly with core count. Stage 0 typically needs 2–5 M steps, so budget an
overnight run per stage on a laptop, or use Kaggle's 30 h/week of session time.

---

## Curriculum

| Stage | Task | API | Advance when |
|---|---|---|---|
| 0 | Stand still | Gymnasium | every eval episode survives 20 s |
| 1 | Stand while being pushed | Gymnasium | >90 % survive 20 s under random shoves |
| 2 | Kick the ball to a target | Gymnasium | ≥50 % deliver the ball, upright |
| 3 | Pass and receive | PettingZoo | ≥40 % complete a pass and trap |
| 4 | Continuous rally | PettingZoo | mean ≥3 consecutive passes |

`python main.py eval --stage N` measures the criterion and prints an explicit
advance / do-not-advance verdict. Do not move on because a reward curve looks nice.

**Honest expectation.** Stages 0–3 are realistic on a laptop. Stage 4 (a sustained rally)
is genuinely research-grade and may need Path B. The code is built for it; convergence is
not promised.

---

## Design

```
PassingSim            physics, observations, reward terms, contact rules (no RL API)
   ├── PassingParallelEnv(ParallelEnv)   PettingZoo — both robots, stages 3–4
   └── SingleAgentEnv(gymnasium.Env)     Gymnasium  — robot 0 only, stages 0–2
```

There is only ever **one scene** — two robots and a ball — and **one core environment**.
Stages differ only in reward weights, initial state and termination. Consequences:

- Observation (84-D) and action (20-D) spaces are **identical in every stage**, so a
  checkpoint loads straight into the next stage with no surgery.
- In stage 0 the ball still exists; it is simply parked 5 m away.
- The Gymnasium env is a *wrapper* that freezes robot 1, not a second simulation.

**Actions are residuals around the standing pose:** `ctrl = q_nominal + action * scale`.
A zero action means "hold the stance". The OP3's zero-joint pose is already stable
(measured: torso at 0.279 m, uprightness 1.000), so the policy starts from a standing
prior instead of from a ragdoll. This one choice does more for training than any
hyperparameter.

Arms are frozen in stages 0–2 by zeroing their action scale while **keeping the action
vector 20-dimensional**, so checkpoints stay compatible. They unfreeze at stage 3 to help
balance — they still may not touch the ball.

### The no-hands rule

OP3 geoms are unnamed (mesh-derived), so contacts are resolved
`contact.geom1/geom2 → model.geom_bodyid → body name`. Any ball contact with
`{l,r}_sho_pitch_link`, `{l,r}_sho_roll_link` or `{l,r}_el_link` is a foul: large negative
reward, and (by default) immediate termination. A penalty alone gets traded away against
other reward terms; termination is what actually teaches the rule.

### Anti-reward-hacking

A pass only counts if the passer stayed upright for `upright_hold_steps` **after** ball
contact. Without that clause the policy learns to fall into the ball — it scores well and
looks terrible.

---

## Tuning notes

Ball mass and friction are the first things to change if kicks are too weak or the ball
flies away (`Config.ball_mass`, `Config.ball_friction`). The ball is deliberately light
(50 g): OP3 servos deliver only 5 N·m, and a regulation ball is simply unkickable.

The OP3's stock feet are narrow (`0.0635 × 0.028 m` half-size), which is the main
stability bottleneck. `Config.widen_feet = True` enlarges the support polygon. It is off
by default so the stock, sim-to-real-credible model stays the baseline — if you turn it
on, say so in your report.

Every reward term is logged separately to TensorBoard (`callbacks.RewardTermLogger`).
When a stage stalls, look there first: the usual cause is one term dominating all others,
which is invisible in the aggregate return.

---

## Troubleshooting

Real problems hit while building this, and their fixes — all already applied in the code:

**Training hangs with the CPUs idle (load average near zero).**
SuperSuit's multiprocessing path deadlocks against MuJoCo: the workers block on a pipe and
nothing progresses. `train.py` passes `num_cpus=0` to `concat_vec_envs_v1`, keeping it
single-threaded. It is also *faster* here (~850 vs ~460 steps/s), because MuJoCo releases
the GIL and the IPC overhead disappears.

**`AttributeError: 'ConcatVecEnv' object has no attribute 'seed'`.**
SB3 ≥2.x calls `VecEnv.seed()` during setup; SuperSuit's `ConcatVecEnv` does not implement
it. `train.SeedShim` supplies it.

**`AttributeError: 'PassingParallelEnv' object has no attribute 'render_mode'`.**
SuperSuit's `MarkovVectorEnv` reads `render_mode` off the raw env. The env sets it.

**Rendering segfaults inside `PPO.load`.** Importing torch and initialising an
OSMesa/EGL context in one interpreter crashes in torch's CUDA bindings, so
`render.py` splits the work across two processes: roll the policy out with torch
(no GL), then replay the recorded trajectory through the renderer (no torch).
A side benefit is that a saved trajectory can be re-rendered at any resolution or
camera angle without re-running the policy. Software rendering of the OP3's
high-poly visual meshes is slow, so `--stride` subsamples frames.

**Rendering dies with an OpenGL error.**
Rendering needs an offscreen GL backend; training and evaluation do not.

```bash
export MUJOCO_GL=egl       # Linux with a GPU (Colab, Kaggle)
export MUJOCO_GL=osmesa    # CPU-only Linux; apt-get install -y libosmesa6 libgl1
export MUJOCO_GL=glfw      # desktop with a display
```

**Stage 0 converges almost immediately.** That is by design, not a bug: the OP3's
zero-joint pose is already stable and actions are residuals around it, so "stand still" is
close to the starting policy. The real balance test is stage 1, under random shoves.

---

## Attribution

The OP3 model comes from Google DeepMind's
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0) and is
fetched at setup rather than vendored. The task design follows DeepMind's
*Learning Agile Soccer Skills for a Bipedal Robot*, which used this same robot.
