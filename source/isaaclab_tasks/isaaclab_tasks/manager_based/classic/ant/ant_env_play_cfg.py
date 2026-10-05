# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""Evaluation (play) configuration of this branch's Ant task.

``Isaac-Ant-Play-v0`` runs the same environment as ``Isaac-Ant-v0`` -- same actions, observations,
sensors and baseline reward definition -- on the sub-terrain that training never sees, at a fixed
difficulty, with the terrain-level curriculum switched off and the foothold scan visualized, as the
group_ten branch does for its play task.

The training classes in ``ant_env_cfg.py`` are intentionally left untouched; this module only adds the
Play variant that the ``Isaac-Ant-Play-v0`` registration in ``__init__.py`` points at.
"""

from isaaclab.utils import configclass

from .ant_env_cfg import EVAL_TERRAINS_CFG, AntEnvCfg


@configclass
class AntEnvCfg_PLAY(AntEnvCfg):
    """Ant task configured for evaluation on the held-out terrain."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 32
        # =======code edit=======
        # Evaluate only on the held-out sub-terrain, at a fixed difficulty, and without the terrain-level
        # curriculum moving robots between rows.
        gen = EVAL_TERRAINS_CFG.copy()
        gen.num_rows, gen.num_cols = 30, 30
        gen.curriculum = False
        gen.difficulty_range = (0.8, 0.8)
        self.scene.terrain.terrain_generator = gen
        # =======code edit=======
        # Show the foothold scan hit points as red spheres (the visualizer prim is /Visuals/FootholdScanner).
        # Play only, as on the group_ten branch: drawing 4096 x 289 markers would slow training down.
        self.scene.foothold_scanner.debug_vis = True
        self.curriculum.terrain_levels = None
        # =======code edit=======
        # There is no domain randomization here to switch off: reset_base uses empty ranges and
        # reset_robot_joints only adds initial-state noise. Uncomment to evaluate from the default pose:
        # self.events.reset_robot_joints = None
