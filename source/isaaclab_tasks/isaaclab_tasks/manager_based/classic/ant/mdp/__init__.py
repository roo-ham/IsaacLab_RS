# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""MDP terms specific to the Ant rough-terrain task."""

from .ant_foot_kinematics import FOOT_NAMES, AntFootKinematics, foot_tip_height  # noqa: F401
from .reward import *  # noqa: F401, F403
