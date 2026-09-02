"""
Builds the two-robot passing scene.

MuJoCo's <include> does not namespace anything, so including op3.xml twice would
collide on every body and actuator name. We therefore compose the scene with
MjSpec, attaching two independently loaded copies of the OP3 under name prefixes
("robot0_", "robot1_"), and write the compiled result to disk.

Generating to a file rather than building in memory on every env construction is
deliberate: the XML stays inspectable in mujoco.viewer, runs are reproducible,
and the MjSpec attach API is exercised in one build step instead of on every run.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

from config import Config

ROBOT_PREFIXES = ("robot0_", "robot1_")

# ── Body groups, resolved by name (OP3 geoms are unnamed / mesh-derived) ──────
TORSO_BODY = "body_link"
FOOT_BODIES = ("l_ank_roll_link", "r_ank_roll_link")
ARM_BODIES = (
    "l_sho_pitch_link", "l_sho_roll_link", "l_el_link",
    "r_sho_pitch_link", "r_sho_roll_link", "r_el_link",
)


# ------------------------------------------------------------------
def _widen_feet(spec: mujoco.MjSpec, halfsize: tuple[float, float, float]) -> int:
    """Enlarge the OP3's narrow foot boxes to grow the support polygon."""
    changed = 0
    for body in spec.bodies:
        if body.name not in FOOT_BODIES:
            continue
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                geom.size = np.array(halfsize)
                changed += 1
    return changed


# ------------------------------------------------------------------
def _add_ball(spec: mujoco.MjSpec, config: Config) -> None:
    """
    A light, well-damped ball. A regulation-weight ball is simply unkickable by
    5 N.m servos, and a bouncy one is far harder to trap, so restitution is kept
    low via solref and rolling friction is enabled with condim=6.
    """
    body = spec.worldbody.add_body(name="ball", pos=list(config.ball_park_pos))
    body.add_freejoint(name="ball_free")
    geom = body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[config.ball_radius, 0, 0],
        mass=config.ball_mass,
        friction=list(config.ball_friction),
        condim=6,
        solref=[0.02, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
        rgba=[0.95, 0.95, 0.95, 1.0],
    )
    geom.priority = 1          # the ball's friction/condim wins every contact pair


# ------------------------------------------------------------------
def _add_world(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_light(
        pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10, 10, 0.05],
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.3, 0.5, 0.3, 1.0],
    )


# ------------------------------------------------------------------
def build_scene(config: Config, spawn_height: float = 0.285) -> Path:
    """
    Compose two prefixed OP3 copies + a ball into one scene and write the XML.

    Robots face each other along x, separated by config.robot_separation.
    Returns the path of the generated file.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "op3_passing"
    spec.option.timestep = 0.002
    # Meshes live in assets/op3/assets; the scene is written to assets/generated.
    spec.meshdir = "../op3/assets"

    _add_world(spec)

    half = config.robot_separation / 2.0
    placements = (
        ([-half, 0.0, spawn_height], [1.0, 0.0, 0.0, 0.0]),          # faces +x
        ([+half, 0.0, spawn_height], [0.0, 0.0, 0.0, 1.0]),          # faces -x (180 deg)
    )
    for prefix, (pos, quat) in zip(ROBOT_PREFIXES, placements):
        child = mujoco.MjSpec.from_file(str(config.op3_xml))
        if config.widen_feet:
            _widen_feet(child, config.foot_halfsize)
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        spec.attach(child, prefix=prefix, frame=frame)

    _add_ball(spec, config)

    model = spec.compile()                      # fail fast on a malformed scene
    config.scene_xml.parent.mkdir(parents=True, exist_ok=True)
    config.scene_xml.write_text(spec.to_xml())
    print(f"[Scene] {config.scene_xml}  nq={model.nq} nu={model.nu} nbody={model.nbody}")
    return config.scene_xml


# ------------------------------------------------------------------
def main() -> int:
    config = Config()
    if not config.op3_xml.exists():
        print(f"  [Error] {config.op3_xml} missing - run: python main.py fetch")
        return 1
    build_scene(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
