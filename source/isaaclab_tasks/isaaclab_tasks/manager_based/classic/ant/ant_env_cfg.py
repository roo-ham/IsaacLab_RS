# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

import math

# =======code edit=======
# The terrain curriculum and terrain-relative height terms live in ant/mdp (non-reward).
import isaaclab_tasks.manager_based.classic.ant.mdp as mymdp
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# =======code edit=======
# Sensors: the height scanner (torso height above the terrain), the wider foothold scanner (where the
# feet land, fed to the policy as the terrain height scan) and the contact sensor (foot air time).
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

##
# Pre-defined configs
##
from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip

# =======code edit=======
# Rough terrain for the Ant task. The held-out sub-terrain is used only by the play configuration, so a
# policy that overfits one sub-terrain shows up in the evaluation numbers. curriculum=True lays the
# difficulty out by row (row i ~ difficulty i/num_rows) so the terrain_levels curriculum below can move
# each robot between rows; with False every tile would get a random difficulty in [0, 1].
HELD_OUT_TERRAIN = "boxes"  # pyramid_stairs, pyramid_stairs_inv, boxes, random_rough, hf_pyramid_slope, hf_pyramid_slope_inv
TRAIN_TERRAINS_CFG = ROUGH_TERRAINS_CFG.replace(
    sub_terrains={name: cfg for name, cfg in ROUGH_TERRAINS_CFG.sub_terrains.items() if name != HELD_OUT_TERRAIN},
    curriculum=True,
)
EVAL_TERRAINS_CFG = ROUGH_TERRAINS_CFG.replace(
    sub_terrains={HELD_OUT_TERRAIN: ROUGH_TERRAINS_CFG.sub_terrains[HELD_OUT_TERRAIN]}
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # terrain
    # =======code edit=======
    # Generated rough terrain, with the same sensors as the group_ten branch: the policy observes the
    # terrain through the foothold scan below, and the height scanner gives the torso height above the
    # terrain rather than world z.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TRAIN_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # =======code edit=======
    # Contact reporting is required by the contact sensor below (foot air time), as on group_ten.
    robot.spawn.activate_contact_sensors = True
    # =======code edit=======
    # Joint PD for position control (ANT_CFG has stiffness=damping=0, i.e. pure torque control). The Ant
    # weighs 0.91 kg and one ankle holds ~1 Nm when standing, while effort control reached 5-15 Nm in
    # kicks: stiffness 20 Nm/rad sags ~0.05 rad under that load and effort_limit 10 Nm (~10x standing
    # torque) leaves room for climbing. Same actuators as the group_ten branch.
    robot.actuators = {
        "body": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=20.0,
            damping=1.0,
            effort_limit=10.0,
        ),
    }
    # =======code edit=======
    # Torso height above the terrain under it (1.2 m x 1.2 m at 0.1 m resolution).
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 1.2]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    # =======code edit=======
    # Wider scan for the policy to see where its feet land: the tips touch down 0.94-1.07 m from the
    # torso, outside the +-0.6 m height_scanner. 2.4 m x 2.4 m at 0.15 m = 17 x 17 = 289 rays. This is
    # the scan the policy observes as terrain_height_scan.
    foothold_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[2.4, 2.4]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/FootholdScanner"),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # =======code edit=======
    # Joint position targets around the default pose (rad): target = default + 0.5 * action. With
    # clip_actions=1.0 that is +-0.5 rad (~29 deg) per joint; the PD actuator turns it into torque.
    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # base_height = ObsTerm(func=mdp.base_pos_z)
        # =======code edit=======
        # Torso height above the terrain, not world z, so it means the same on every sub-terrain.
        base_height = ObsTerm(func=mymdp.torso_height_obs, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )

        # =======code edit=======
        # Terrain scan supplies ground-relative height: 289 values from the foothold scanner, the only
        # observation that sees the terrain itself (same term and sensor as the group_ten branch).
        terrain_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foothold_scanner"), "offset": 0.0},
            clip=(-1.0, 1.0),
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # =======code edit=======
    # Domain randomization: a discrete, shared friction value per environment, in exact 0.1 increments,
    # assigned once at startup (same scenario as the group_ten branch).
    discrete_friction = EventTerm(
        func=mymdp.randomize_discrete_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_values": (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    # (6) Penalty for energy consumption
    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    # (7) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    # torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})
    # =======code edit=======
    # World z does not describe a fall on generated terrain: some sub-terrains (e.g. inverted pyramids)
    # sit below z = 0 and the robot stands higher on a step, so the fixed 0.31 m threshold would reset
    # robots every step. Use the terrain-independent orientation check instead.
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(50.0)})


# =======code edit=======
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Moves each robot between terrain rows from the episode that just ended: one row up when it reached
    # the time-out without falling at >= 1.5 m/s average forward speed, one row down when it fell or
    # averaged < 0.3 m/s. Logged as Curriculum/terrain_levels (mean level).
    terrain_levels = CurrTerm(func=mymdp.terrain_levels_speed, params={"up_speed": 1.5, "down_speed": 0.3})


@configclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Ant walking environment."""

    # Scene settings
    # Contact sensors need real USD prims for every env; fabric cloning only creates env_0 in USD.
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # =======code edit=======
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # =======code edit=======
        # The generated terrain brings its own material (multiply combine modes); keep the simulation
        # default aligned with it.
        self.sim.physics_material = self.scene.terrain.physics_material
        # =======code edit=======
        # Tick the sensors at the rates they need (same as the group_ten branch): the scanners once per
        # control step, the contact sensor at the physics rate.
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.foothold_scanner is not None:
            self.scene.foothold_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
