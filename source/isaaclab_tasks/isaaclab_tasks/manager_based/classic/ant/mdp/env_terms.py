# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""Non-reward MDP terms for the Ant rough-terrain task (terrain curriculum).

The body of :func:`terrain_levels_speed` is copied verbatim from the group_ten branch, where the rough
terrain and its curriculum were set up. It is not a reward function and reads no sensor: it only moves
each robot between terrain rows from the episode that just ended.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

__all__ = ["terrain_levels_speed"]


# =======code edit=======
def terrain_levels_speed(
    env, env_ids, up_speed: float, down_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terrain-level curriculum from the average forward (+x) speed of the episode that just ended.

    Runs at reset, before the scene is reset, so position, episode length and the termination flags still
    describe the finished episode. A robot moves one row up (harder) if it reached the time-out without
    falling and averaged at least up_speed (m/s), and one row down (easier) if it fell or averaged less
    than down_speed. Same idea as terrain_levels_vel, which needs a velocity command this task does not use.

    Returns:
        The mean terrain level over all robots.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    distance = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    elapsed = (env.episode_length_buf[env_ids] * env.step_dt).clamp(min=env.step_dt)
    avg_speed = distance / elapsed
    fell = env.termination_manager.terminated[env_ids]
    timed_out = env.termination_manager.time_outs[env_ids]
    move_up = timed_out & ~fell & (avg_speed >= up_speed)
    move_down = (fell | (avg_speed < down_speed)) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
