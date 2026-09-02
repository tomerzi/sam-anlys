# Working with Claude on this project

Notes for driving this project like a senior engineer rather than a prompt-and-hope
student. RL is unusually punishing of sloppy workflow: runs are long, failures are silent,
and a bad reward term can waste a week of compute.

## Which model for what

| Use | Model | Why |
|---|---|---|
| Architecture, reward design, "why won't this learn?" | **Opus** | The expensive mistakes here are conceptual, not typographical. A wrong reward term costs days. |
| Writing modules once the design is fixed, refactors, docstrings, plotting | **Sonnet** | Much cheaper and fast enough. The design is already decided. |

Use **plan mode** before touching RL code — reward functions are cheap to write and
expensive to debug.

## The habits that actually matter

1. **Never accept a reward function you cannot read line by line.** You will be tuning it
   for weeks and defending it in a report. It has to be yours.
2. **Commit every working checkpoint before you change the reward.** RL regressions are
   silent and otherwise irreversible.
3. **Make Claude diagnose before it fixes.** "Why is `ep_len_mean` flat at 50?" produces a
   diagnosis. "Fix my training" produces a random walk through hyperparameters.
4. **One variable per experiment**, and put it in the run name:
   `stage2_ball05kg_lr3e4`. Two changes at once tells you nothing.
5. **Watch the video, not the curve.** `python main.py render` after every stage. Every
   high-reward humanoid failure — shuffling, diving at the ball, vibrating in place —
   looks perfectly healthy in TensorBoard.
6. **Run `python main.py check` before every training job.** Ten seconds against hours.
7. **Give Claude the context, not the whole repo.** `CLAUDE.md` at the repo root already
   states the conventions and the invariants, so you do not have to re-explain them each
   session.

## Debugging a stage that will not learn

Work down this list; it is roughly ordered by how often each is the real cause.

1. **Look at the per-term reward plots** (`reward/*` in TensorBoard). One term dominating
   the rest is the single most common failure, and it is invisible in `ep_rew_mean`.
2. **Render a rollout.** Is it doing something dumb but high-scoring?
3. **Check episode length.** Falling immediately (`ep_len_mean` ≈ 50) means the posture
   terms or `action_scale` are wrong, not the task reward.
4. **Sanity-check the shaping sign.** A potential-shaping term should be ~0 at reset, not
   a large negative number. `check.py` asserts this.
5. **Only then touch hyperparameters.** Learning rate and entropy are the last resort, not
   the first.

## What to be honest about in your report

Stages 0–3 are achievable on a laptop. A sustained rally (stage 4) is research-grade work
that DeepMind did with far more compute. Saying so — and showing the curriculum, the
measured per-stage success criteria, and where it plateaued — is a stronger project than
overclaiming a result you did not get.
