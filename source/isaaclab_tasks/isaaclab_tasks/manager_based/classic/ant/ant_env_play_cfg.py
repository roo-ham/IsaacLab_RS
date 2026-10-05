# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""Evaluation (play) configuration of this branch's Ant task.

``Isaac-Ant-Play-v0`` runs exactly the same environment as ``Isaac-Ant-v0`` -- plane terrain, joint
effort actions, the same observations and the same baseline reward definition -- with the evaluation
settings the group_ten branch uses for its play task: a small scene and no curriculum or randomization
that would change between episodes.

The baseline classes in ``ant_env_cfg.py`` are intentionally left untouched; this module only adds the
Play variant that the ``Isaac-Ant-Play-v0`` registration in ``__init__.py`` points at.
"""

from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg


@configclass
class AntEnvCfg_PLAY(AntEnvCfg):
    """Baseline Ant task configured for evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 32
        # The baseline task has no domain randomization to switch off: reset_base uses empty ranges and
        # reset_robot_joints only adds initial-state noise. Uncomment to evaluate from the default pose:
        # self.events.reset_robot_joints = None
