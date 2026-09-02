"""
Core simulation for the two-robot passing task.

This class owns the MuJoCo model/data, the reset distributions, the observation
assembly and every contact query. It deliberately exposes no RL API: the
Gymnasium and PettingZoo adapters in envs/ are thin wrappers over it, so reward
and termination logic exists in exactly one place.
"""
import numpy as np
import mujoco

from config import Config
from scene_builder import ROBOT_PREFIXES, TORSO_BODY, FOOT_BODIES, ARM_BODIES

N_JOINTS = 20
OBS_DIM = 84

# Actuator slots within a robot's 20-vector (order comes from op3.xml).
HEAD_SLOTS = (0, 1)
ARM_SLOTS = (2, 3, 4, 5, 6, 7)
LEG_SLOTS = tuple(range(8, 20))

# Joints an OP3 needs bent to stand; zeros are already a stable stance, so the
# nominal pose is all-zeros and a zero action means "hold the standing pose".
NOMINAL_QPOS = np.zeros(N_JOINTS, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
class RobotIndex:
    """Resolved model indices for one robot. Built once, at construction."""

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.prefix = prefix
        name2id = lambda t, s: mujoco.mj_name2id(model, t, s)

        self.torso_bid = name2id(mujoco.mjtObj.mjOBJ_BODY, prefix + TORSO_BODY)
        # The freejoint is unnamed after attachment; reach it through the torso.
        free_jid = model.body_jntadr[self.torso_bid]
        self.free_qadr = model.jnt_qposadr[free_jid]
        self.free_dadr = model.jnt_dofadr[free_jid]

        joint_names = [
            "head_pan", "head_tilt",
            "l_sho_pitch", "l_sho_roll", "l_el",
            "r_sho_pitch", "r_sho_roll", "r_el",
            "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
            "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
        ]
        jids = [name2id(mujoco.mjtObj.mjOBJ_JOINT, prefix + j) for j in joint_names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.act_ids = np.array([
            name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + j + "_act") for j in joint_names
        ])

        self.foot_bids = [name2id(mujoco.mjtObj.mjOBJ_BODY, prefix + b) for b in FOOT_BODIES]
        # OP3 geoms are unnamed (mesh-derived), so groups resolve via geom_bodyid.
        arm_bids = {name2id(mujoco.mjtObj.mjOBJ_BODY, prefix + b) for b in ARM_BODIES}
        self.arm_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in arm_bids}
        self.foot_geoms = {g for g in range(model.ngeom)
                           if model.geom_bodyid[g] in set(self.foot_bids)}
        body_geoms = {g for g in range(model.ngeom)
                      if model.geom_bodyid[g] == self.torso_bid}
        self.legal_geoms = (self.foot_geoms | body_geoms) - self.arm_geoms


# ══════════════════════════════════════════════════════════════════════════════
class PassingSim:
    """MuJoCo physics + observations for two OP3 robots and a ball."""

    def __init__(self, config: Config, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(config.scene_xml))
        self.data = mujoco.MjData(self.model)

        self.robots = [RobotIndex(self.model, p) for p in ROBOT_PREFIXES]
        # geom -> owning robot index (-1 for world, floor and ball).
        self.geom_owner = np.full(self.model.ngeom, -1, dtype=np.int64)
        for i, prefix in enumerate(ROBOT_PREFIXES):
            for g in range(self.model.ngeom):
                bid = self.model.geom_bodyid[g]
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if name.startswith(prefix):
                    self.geom_owner[g] = i

        self.ball_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        ball_jid = self.model.body_jntadr[self.ball_bid]
        self.ball_qadr = self.model.jnt_qposadr[ball_jid]
        self.ball_dadr = self.model.jnt_dofadr[ball_jid]

        # Per-slot action scale; zeroing the arm slots freezes them without
        # changing the action dimensionality, so checkpoints stay compatible.
        self.action_scale = np.full(N_JOINTS, config.action_scale)
        if config.stage < 3:
            self.action_scale[list(ARM_SLOTS)] = config.arm_action_scale

        self.prev_action = np.zeros((2, N_JOINTS))
        self.step_count = 0

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self, passer: int = 0) -> None:
        cfg = self.config
        mujoco.mj_resetData(self.model, self.data)
        self.passer = passer
        self.step_count = 0
        self.prev_action[:] = 0.0

        half = cfg.robot_separation / 2.0
        for i, r in enumerate(self.robots):
            sign = -1.0 if i == 0 else 1.0
            self.data.qpos[r.free_qadr:r.free_qadr + 3] = [sign * half, 0.0, 0.285]
            quat = [1.0, 0.0, 0.0, 0.0] if i == 0 else [0.0, 0.0, 0.0, 1.0]
            self.data.qpos[r.free_qadr + 3:r.free_qadr + 7] = quat
            noise = cfg.joint_noise if cfg.stage >= 1 else 0.0
            self.data.qpos[r.qadr] = NOMINAL_QPOS + self.rng.normal(0, noise, N_JOINTS)

        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = self._initial_ball_pos()
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

    def _initial_ball_pos(self) -> np.ndarray:
        """Parked far away until the ball becomes part of the task (stage 2)."""
        cfg = self.config
        if cfg.stage < 2:
            return np.array(cfg.ball_park_pos)
        # Just in front of the passer's feet, with a little spread.
        sign = -1.0 if self.passer == 0 else 1.0
        x = sign * (cfg.robot_separation / 2.0 - 0.20)
        return np.array([x + self.rng.uniform(-0.03, 0.03),
                         self.rng.uniform(-0.05, 0.05),
                         cfg.ball_radius])

    # ── Step ──────────────────────────────────────────────────────────────
    def step(self, actions: np.ndarray) -> None:
        """actions: (2, 20) in [-1, 1], as residuals around the standing pose."""
        cfg = self.config
        for i, r in enumerate(self.robots):
            target = NOMINAL_QPOS + np.clip(actions[i], -1.0, 1.0) * self.action_scale
            self.data.ctrl[r.act_ids] = target

        if cfg.stage >= 1 and self.step_count % cfg.push_interval_steps == 0 \
                and self.step_count > 0:
            self._apply_push()

        for _ in range(cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.data.xfrc_applied[:] = 0.0
        self.prev_action[:] = actions
        self.step_count += 1

    def _apply_push(self) -> None:
        """Random horizontal shove on each torso - the balance curriculum."""
        for r in self.robots:
            direction = self.rng.normal(size=2)
            direction /= np.linalg.norm(direction) + 1e-8
            self.data.xfrc_applied[r.torso_bid, :2] = direction * self.config.push_force

    # ── State queries ─────────────────────────────────────────────────────
    def torso_pos(self, i: int) -> np.ndarray:
        return self.data.xpos[self.robots[i].torso_bid].copy()

    def torso_mat(self, i: int) -> np.ndarray:
        return self.data.xmat[self.robots[i].torso_bid].reshape(3, 3)

    def uprightness(self, i: int) -> float:
        """1.0 fully upright, 0.0 horizontal, negative upside down."""
        return float(self.torso_mat(i)[2, 2])

    def torso_vel(self, i: int) -> np.ndarray:
        r = self.robots[i]
        return self.data.qvel[r.free_dadr:r.free_dadr + 3].copy()

    def torso_angvel(self, i: int) -> np.ndarray:
        r = self.robots[i]
        return self.data.qvel[r.free_dadr + 3:r.free_dadr + 6].copy()

    def joint_pos(self, i: int) -> np.ndarray:
        return self.data.qpos[self.robots[i].qadr].copy()

    def joint_vel(self, i: int) -> np.ndarray:
        return self.data.qvel[self.robots[i].dadr].copy()

    def actuator_force(self, i: int) -> np.ndarray:
        return self.data.actuator_force[self.robots[i].act_ids].copy()

    def ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_bid].copy()

    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self.ball_dadr:self.ball_dadr + 3].copy()

    def foot_contacts(self, i: int) -> np.ndarray:
        """Boolean per foot: is it touching anything?"""
        r = self.robots[i]
        out = np.zeros(2, dtype=np.float64)
        for c in self.data.contact[:self.data.ncon]:
            for k, bid in enumerate(r.foot_bids):
                geoms = {g for g in (c.geom1, c.geom2)}
                if any(self.model.geom_bodyid[g] == bid for g in geoms):
                    out[k] = 1.0
        return out

    def fallen(self, i: int) -> bool:
        cfg = self.config
        return (self.torso_pos(i)[2] < cfg.min_torso_height
                or self.uprightness(i) < cfg.min_uprightness)

    # ── Contact rules: the no-hands constraint ────────────────────────────
    def _ball_contact_geoms(self) -> set[int]:
        """Every geom currently touching the ball."""
        touching = set()
        for c in self.data.contact[:self.data.ncon]:
            if c.geom1 == self.ball_gid:
                touching.add(int(c.geom2))
            elif c.geom2 == self.ball_gid:
                touching.add(int(c.geom1))
        return touching

    def hand_foul(self, i: int) -> bool:
        """True when the ball touches this robot's hand/arm - always illegal."""
        return bool(self._ball_contact_geoms() & self.robots[i].arm_geoms)

    def any_hand_foul(self) -> tuple[bool, bool]:
        touching = self._ball_contact_geoms()
        return (bool(touching & self.robots[0].arm_geoms),
                bool(touching & self.robots[1].arm_geoms))

    def legal_ball_contact(self, i: int) -> bool:
        """Ball touched by this robot's feet, legs, torso or head - a legal touch."""
        mine = {g for g in self._ball_contact_geoms() if self.geom_owner[g] == i}
        return bool(mine - self.robots[i].arm_geoms)

    # ── Observation ───────────────────────────────────────────────────────
    def observe(self, i: int) -> np.ndarray:
        """
        84-D egocentric observation, identical in shape across every curriculum
        stage so a checkpoint from stage N loads straight into stage N+1.
        """
        other = 1 - i
        rot = self.torso_mat(i)
        pos = self.torso_pos(i)
        to_local = lambda v: rot.T @ v

        ball_rel = to_local(self.ball_pos() - pos)
        ball_vel_rel = to_local(self.ball_vel())
        mate_rel = to_local(self.torso_pos(other) - pos)

        mate_fwd = self.torso_mat(other)[:, 0]
        mate_fwd_local = to_local(mate_fwd)
        heading = np.array([mate_fwd_local[0], mate_fwd_local[1]])
        norm = np.linalg.norm(heading)
        heading = heading / norm if norm > 1e-8 else np.array([1.0, 0.0])

        role = np.array([1.0, 0.0]) if i == self.passer else np.array([0.0, 1.0])

        return np.concatenate([
            self.joint_pos(i),                    # 20
            self.joint_vel(i),                    # 20
            to_local(np.array([0.0, 0.0, -1.0])),  # 3  gravity in torso frame
            self.torso_angvel(i),                 # 3
            to_local(self.torso_vel(i)),          # 3
            self.prev_action[i],                  # 20
            ball_rel,                             # 3
            ball_vel_rel,                         # 3
            mate_rel,                             # 3
            heading,                              # 2
            role,                                 # 2
            self.foot_contacts(i),                # 2
        ]).astype(np.float64)
