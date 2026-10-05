# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""Non-reward MDP terms for the Ant rough-terrain task.

Copied from the group_ten branch (``ant/mdp/reward.py``), where the terrain was set up. Only the term
the terrain itself needs is kept here: no reward functions, and no sensor-based terms, because this
branch trains the Ant without any sensors in the scene.
"""

from .env_terms import *  # noqa: F401, F403
