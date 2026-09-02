"""
Smoke test - the cheapest way to catch the bugs that waste training runs.

Run this before every training job. It verifies that the scene loads, that the
observation contract holds, that the robots actually stand under a zero action,
and that the no-hands rule fires on arms but not on feet.
"""
import sys

import mujoco
import numpy as np

from config import Config
from sim.passing_sim import PassingSim, OBS_DIM, N_JOINTS
from sim.rewards import new_pass_state, compute_rewards
from envs.single_env import SingleAgentEnv
from envs.parallel_env import PassingParallelEnv


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


# ------------------------------------------------------------------
def run_checks(config: Config) -> bool:
    print(f"\n{'='*60}\n[Check] smoke test\n{'='*60}")
    ok = True

    if not config.scene_xml.exists():
        print(f"  [FAIL] {config.scene_xml} missing - run: python main.py build")
        return False

    sim = PassingSim(config)
    sim.reset()
    ok &= _report("scene loads", True, f"nq={sim.model.nq} nu={sim.model.nu}")
    ok &= _report("observation is 84-D", sim.observe(0).shape == (OBS_DIM,),
                  f"got {sim.observe(0).shape}")

    # Zero action must hold the standing pose. If this fails, the nominal pose is
    # wrong and no amount of RL will fix it.
    sim.reset()
    for _ in range(500):
        sim.step(np.zeros((2, N_JOINTS)))
    heights = [sim.torso_pos(i)[2] for i in range(2)]
    uprights = [sim.uprightness(i) for i in range(2)]
    standing = all(h > config.min_torso_height for h in heights) and \
        all(u > 0.9 for u in uprights)
    ok &= _report("robots stand under zero action (10 s)", standing,
                  f"z={np.round(heights,3)} upright={np.round(uprights,3)}")

    finite = np.all(np.isfinite(sim.observe(0))) and np.all(np.isfinite(sim.data.qpos))
    ok &= _report("no NaNs after 500 steps", bool(finite))

    # The no-hands rule: must fire on an arm, must not fire on a foot.
    def ball_at(body: str):
        sim.reset()
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body)
        sim.data.qpos[sim.ball_qadr:sim.ball_qadr + 3] = sim.data.xpos[bid]
        mujoco.mj_forward(sim.model, sim.data)
        for _ in range(3):
            mujoco.mj_step(sim.model, sim.data)

    ball_at("robot0_l_el_link")
    ok &= _report("hand contact flagged as a foul", sim.hand_foul(0))
    ball_at("robot0_l_ank_roll_link")
    ok &= _report("foot contact is legal", (not sim.hand_foul(0)) and sim.legal_ball_contact(0))

    # Reward wiring, including the potential-shaping initialisation.
    stage2 = Config(stage=2)
    sim2 = PassingSim(stage2)
    sim2.reset()
    state = new_pass_state(sim2)
    zeros = np.zeros((2, N_JOINTS))
    _, terms = compute_rewards(sim2, state, zeros, zeros)
    shaping_ok = abs(terms[0].get("ball_progress", 0.0)) < 1e-6
    ok &= _report("ball_progress starts at zero", shaping_ok,
                  f"got {terms[0].get('ball_progress', 0.0):+.4f}")

    # Both adapters agree on the interface.
    gym_env = SingleAgentEnv(config)
    o, _ = gym_env.reset(seed=0)
    ok &= _report("Gymnasium env steps", gym_env.step(np.zeros(N_JOINTS))[0].shape == (OBS_DIM,))

    pz = PassingParallelEnv(Config(stage=3))
    obs, _ = pz.reset(seed=0)
    acts = {a: np.zeros(N_JOINTS, dtype=np.float32) for a in pz.agents}
    obs, rew, term, trunc, _ = pz.step(acts)
    ok &= _report("PettingZoo env steps", len(obs) == 2 and len(rew) == 2)
    ok &= _report("both agents terminate together",
                  len(set(term.values())) == 1 and len(set(trunc.values())) == 1)

    print(f"{'='*60}\n[Check] {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}\n{'='*60}\n")
    return ok


def main() -> int:
    return 0 if run_checks(Config()) else 1


if __name__ == "__main__":
    sys.exit(main())
