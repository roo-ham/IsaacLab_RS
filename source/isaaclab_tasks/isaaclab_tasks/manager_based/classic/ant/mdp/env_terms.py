# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""Non-reward MDP terms for the Ant rough-terrain task.

The bodies of these terms are copied verbatim from the group_ten branch, where the rough terrain, its
curriculum, the domain randomization and the sensor-based observations were set up. None of them is a
reward function: :func:`torso_height_obs` is the terrain-relative height observation (height scanner),
:func:`terrain_levels_speed` is the terrain curriculum and :func:`randomize_discrete_friction` is the
friction randomization.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

__all__ = [
    "randomize_discrete_friction",
    "terrain_levels_speed",
    "torso_height_above_ground",
    "torso_height_obs",
]


# =======code edit=======
def torso_height_above_ground(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso height above the terrain under it, shape (num_envs,).

    The ground level is the mean of the height-scanner hits; rays that missed the terrain (inf) are
    ignored. World z alone is wrong on generated terrain, whose sub-terrains (e.g. inverted pyramids)
    can sit below z = 0.
    """
    scanner = env.scene.sensors[sensor_cfg.name]
    hits_z = scanner.data.ray_hits_w[..., 2]
    finite = torch.isfinite(hits_z)
    ground_z = torch.where(finite, hits_z, 0.0).sum(dim=1) / finite.sum(dim=1).clamp(min=1)
    return scanner.data.pos_w[:, 2] - ground_z


def torso_height_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso height above the terrain as a (num_envs, 1) observation."""
    return torso_height_above_ground(env, sensor_cfg).unsqueeze(-1)


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


# =======code edit=======
def randomize_discrete_friction(
    env, env_ids, asset_cfg: SceneEntityCfg, friction_values: tuple[float, ...]
) -> None:
    """Assign one discrete, shared friction value to every Ant collider in each environment."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    asset = env.scene[asset_cfg.name]
    values = torch.tensor(friction_values, dtype=torch.float32, device="cpu")
    value_ids = torch.randint(len(values), (len(env_ids),), device="cpu")
    friction = values[value_ids]
    material_samples = torch.stack((friction, friction, torch.zeros_like(friction)), dim=-1)

    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids] = material_samples.unsqueeze(1).expand(-1, materials.shape[1], -1)
    asset.root_physx_view.set_material_properties(materials, env_ids)
