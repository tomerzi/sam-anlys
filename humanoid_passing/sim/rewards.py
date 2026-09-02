"""
Reward terms for the passing task.

Every term is a small function returning a scalar, and compute_rewards returns
both the total and a per-term breakdown. The breakdown is not decoration: the
usual failure mode of a humanoid reward is one term quietly dominating all the
others, and that is invisible in the aggregate return. callbacks.RewardTermLogger
writes these to TensorBoard.
"""
from dataclasses import dataclass, field

import numpy as np

from config import Config
from sim.passing_sim import PassingSim

NOMINAL_HEIGHT = 0.279     # measured: OP3 torso height in the zero-joint stance


# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PassState:
    """Per-episode bookkeeping that the reward needs but physics does not hold."""
    prev_ball_dist: float = 0.0
    last_toucher: int = -1
    touch_step: int = -1
    upright_since_touch: int = 0
    passes: int = 0
    delivered: bool = False
    fouled: tuple[bool, bool] = (False, False)
    terms: list[dict[str, float]] = field(default_factory=list)


# ── Posture terms (every stage) ──────────────────────────────────────────────
def r_alive(sim: PassingSim, i: int) -> float:
    return 0.0 if sim.fallen(i) else 1.0


def r_height(sim: PassingSim, i: int) -> float:
    """Gaussian around the natural standing height - tolerant of a small crouch."""
    dz = sim.torso_pos(i)[2] - NOMINAL_HEIGHT
    return float(np.exp(-(dz / 0.06) ** 2))


def r_upright(sim: PassingSim, i: int) -> float:
    return float(np.clip(sim.uprightness(i), 0.0, 1.0))


def p_com_vel(sim: PassingSim, i: int) -> float:
    """Penalise drifting. The task is to stand and pass, not to wander."""
    return float(np.sum(sim.torso_vel(i)[:2] ** 2))


def p_joint_vel(sim: PassingSim, i: int) -> float:
    return float(np.sum(sim.joint_vel(i) ** 2))


def p_action_rate(action: np.ndarray, prev_action: np.ndarray) -> float:
    return float(np.sum((action - prev_action) ** 2))


def p_torque(sim: PassingSim, i: int) -> float:
    return float(np.sum(sim.actuator_force(i) ** 2))


# ── Ball terms (stage 2+) ────────────────────────────────────────────────────
def target_pos(sim: PassingSim, passer: int) -> np.ndarray:
    """A pass is aimed at the receiver's feet."""
    p = sim.torso_pos(1 - passer).copy()
    p[2] = sim.config.ball_radius
    return p


def r_ball_progress(sim: PassingSim, state: PassState, passer: int) -> tuple[float, float]:
    """Potential shaping: reward reduction in ball -> target distance."""
    dist = float(np.linalg.norm(sim.ball_pos()[:2] - target_pos(sim, passer)[:2]))
    return state.prev_ball_dist - dist, dist


def r_ball_speed(sim: PassingSim, passer: int) -> float:
    """Ball velocity projected onto the direction of the receiver, clipped >= 0."""
    to_target = target_pos(sim, passer)[:2] - sim.ball_pos()[:2]
    norm = np.linalg.norm(to_target)
    if norm < 1e-6:
        return 0.0
    return float(max(0.0, np.dot(sim.ball_vel()[:2], to_target / norm)))


def r_trap(sim: PassingSim, receiver: int) -> float:
    """Receiver is rewarded for the ball being close AND slow - a controlled trap."""
    cfg = sim.config
    dist = float(np.linalg.norm(sim.ball_pos()[:2] - sim.torso_pos(receiver)[:2]))
    if dist > cfg.arrival_radius:
        return 0.0
    speed = float(np.linalg.norm(sim.ball_vel()[:2]))
    return float(np.clip(1.0 - speed / max(cfg.trap_speed, 1e-6), 0.0, 1.0))


# ------------------------------------------------------------------
def new_pass_state(sim: PassingSim) -> PassState:
    """
    Fresh bookkeeping for an episode.

    prev_ball_dist MUST start at the true initial distance: leaving it at 0.0
    makes the first potential-shaping term equal to -dist, handing every episode
    a large spurious penalty on step one.
    """
    state = PassState()
    if sim.config.stage >= 2:
        state.prev_ball_dist = float(np.linalg.norm(
            sim.ball_pos()[:2] - target_pos(sim, sim.passer)[:2]))
    return state


# ══════════════════════════════════════════════════════════════════════════════
def compute_rewards(
    sim: PassingSim,
    state: PassState,
    actions: np.ndarray,
    prev_actions: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    Returns per-agent rewards (2,) and a per-agent term breakdown.

    The passer is rewarded for moving the ball toward the receiver; the receiver
    for trapping it. Both are always rewarded for staying upright, which is what
    keeps the robots standing rather than diving at the ball.
    """
    cfg = sim.config
    passer = sim.passer
    rewards = np.zeros(2)
    breakdown: list[dict[str, float]] = []

    fouls = sim.any_hand_foul()
    state.fouled = fouls

    progress, dist = (0.0, state.prev_ball_dist)
    if cfg.stage >= 2:
        progress, dist = r_ball_progress(sim, state, passer)

    for i in range(2):
        terms: dict[str, float] = {
            "alive": cfg.w_alive * r_alive(sim, i),
            "height": cfg.w_height * r_height(sim, i),
            "upright": cfg.w_upright * r_upright(sim, i),
            "com_vel": cfg.w_com_vel * p_com_vel(sim, i),
            "joint_vel": cfg.w_joint_vel * p_joint_vel(sim, i),
            "action_rate": cfg.w_action_rate * p_action_rate(actions[i], prev_actions[i]),
            "torque": cfg.w_torque * p_torque(sim, i),
        }

        if cfg.stage >= 2:
            if i == passer:
                terms["ball_progress"] = cfg.w_ball_progress * progress
                terms["ball_speed"] = cfg.w_ball_speed * r_ball_speed(sim, passer)
            else:
                terms["trap"] = cfg.w_trap * r_trap(sim, i)

            if fouls[i]:
                terms["hand_foul"] = cfg.w_hand_foul

        rewards[i] = sum(terms.values())
        breakdown.append(terms)

    state.prev_ball_dist = dist
    return rewards, breakdown


# ------------------------------------------------------------------
def update_pass_state(sim: PassingSim, state: PassState) -> bool:
    """
    Track legal touches and completed passes.

    A pass only counts if the passer was still upright for upright_hold_steps
    after the touch. Without that clause the policy learns to fall into the ball,
    which scores well and looks terrible.
    """
    cfg = sim.config
    completed = False

    for i in range(2):
        if sim.legal_ball_contact(i):
            if state.last_toucher != i:
                state.last_toucher = i
                state.touch_step = sim.step_count
                state.upright_since_touch = 0

    if state.last_toucher >= 0 and not sim.fallen(state.last_toucher):
        state.upright_since_touch += 1

    receiver = 1 - sim.passer
    if (state.last_toucher == sim.passer
            and state.upright_since_touch >= cfg.upright_hold_steps
            and r_trap(sim, receiver) > 0.5):
        state.passes += 1
        state.delivered = True
        completed = True

    return completed
