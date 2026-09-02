"""
Configuration for the humanoid ball-passing RL project.

A single dataclass holds every tunable, mirroring the convention used by the
football_action_coach package. Instantiate it once, in main().
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Paths (relative: run everything from inside humanoid_passing/) ---
    op3_xml: Path = Path("assets/op3/op3.xml")
    scene_xml: Path = Path("assets/generated/passing_scene.xml")
    runs_dir: Path = Path("runs")

    # --- Curriculum stage ---
    # 0 stand | 1 stand under push | 2 kick to target | 3 pass + receive | 4 rally
    stage: int = 0

    # --- Scene layout ---
    robot_separation: float = 1.0        # metres between the two robots
    widen_feet: bool = False             # enlarge the support polygon (see README)
    foot_halfsize: tuple[float, float, float] = (0.075, 0.045, 0.006)

    # --- Ball (sized for a 0.5 m robot driven by 5 N.m actuators) ---
    ball_radius: float = 0.05
    ball_mass: float = 0.05
    ball_friction: tuple[float, float, float] = (0.6, 0.005, 0.0002)
    ball_park_pos: tuple[float, float, float] = (5.0, 0.0, 0.05)   # stage 0-1 parking spot
    ball_bounds: float = 4.0             # episode ends if the ball leaves this radius

    # --- Control ---
    frame_skip: int = 10                 # 0.002 s * 10 -> 50 Hz control
    episode_seconds: float = 20.0
    action_scale: float = 0.5            # radians, residual around the standing pose
    arm_action_scale: float = 0.0        # 0 freezes the arms; unfrozen from stage 3

    # --- Termination ---
    min_torso_height: float = 0.20
    min_uprightness: float = 0.5         # projected gravity, 1.0 = perfectly upright
    hand_contact_terminates: bool = True  # the no-hands rule (see README)

    # --- Reward weights: posture (all stages) ---
    w_alive: float = 1.0
    w_height: float = 1.0
    w_upright: float = 1.0
    w_com_vel: float = -0.5              # penalise drifting
    w_joint_vel: float = -1e-3
    w_action_rate: float = -1e-2
    w_torque: float = -1e-4

    # --- Reward weights: ball (stages 2+) ---
    w_ball_progress: float = 3.0         # potential shaping on ball -> target distance
    w_ball_speed: float = 1.0            # ball velocity projected at the target
    w_arrival: float = 10.0              # one-off bonus when the ball reaches the target
    w_trap: float = 10.0                 # receiver damps the ball near itself
    w_pass: float = 20.0                 # a completed, legal pass (stage 4 rally)
    w_hand_foul: float = -5.0            # ball touched by hand/arm

    # --- Pass bookkeeping ---
    arrival_radius: float = 0.30         # ball counts as delivered within this radius
    trap_speed: float = 0.25             # ball counts as trapped below this speed
    upright_hold_steps: int = 25         # passer must stay upright this long after contact

    # --- Domain randomisation (stage 1+) ---
    push_interval_steps: int = 100
    push_force: float = 8.0              # newtons applied to the torso
    joint_noise: float = 0.05            # radians, added at reset
    friction_range: tuple[float, float] = (0.7, 1.3)

    # --- PPO ---
    n_envs: int = 8
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    # Initial action-noise std = exp(log_std_init). The default SB3 value of 0.0
    # (std 1.0) drives the joints to their full +-action_scale immediately and
    # topples the robot before it can learn anything, so start quieter.
    log_std_init: float = -1.0

    # --- Misc ---
    seed: int = 0
    device: str = "cpu"                  # SB3 + MuJoCo is CPU-bound; see README

    # ------------------------------------------------------------------
    @property
    def control_dt(self) -> float:
        return 0.002 * self.frame_skip

    @property
    def max_episode_steps(self) -> int:
        return int(self.episode_seconds / self.control_dt)

    def stage_dir(self, stage: int | None = None) -> Path:
        return self.runs_dir / f"stage{self.stage if stage is None else stage}"
