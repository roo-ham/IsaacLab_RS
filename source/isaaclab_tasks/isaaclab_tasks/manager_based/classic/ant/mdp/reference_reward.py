# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""The Ant reward definition of the reference (cailab) version, frozen for evaluation.

The task rewards in ``ant_env_cfg.py`` are retuned on this branch, so an episode reward total
computed with them means something different from commit to commit. Evaluation therefore reports the
total with the reward functions and weights of the reference version -- the Ant ``RewardsCfg`` at
commit e83a5d2f11ca1b5f03b690e1978479e620c500e2 (``upstream/main``), copied below verbatim:

    term              weight   function                                    parameters
    progress           1.0     humanoid.mdp.progress_reward                target_pos (1000, 0, 0)
    alive              0.5     isaaclab.envs.mdp.is_alive
    upright            0.1     humanoid.mdp.upright_posture_bonus          threshold 0.93
    move_to_target     0.5     humanoid.mdp.move_to_target_bonus           threshold 0.8, target_pos (1000, 0, 0)
    action_l2         -0.005   isaaclab.envs.mdp.action_l2
    energy            -0.05    humanoid.mdp.power_consumption              gear_ratio {".*": 15.0}
    joint_pos_limits  -0.1     humanoid.mdp.joint_pos_limits_penalty_ratio threshold 0.99, gear_ratio {".*": 15.0}

Only the weights and parameters are copied: the functions themselves still exist unchanged (the
humanoid mdp package and the core isaaclab mdp package are untouched on this branch relative to that
commit), so importing them reproduces the reference values exactly as the reference accounting does,
i.e. each step contributes ``term * weight * step_dt`` and an episode is their sum.

``play_one_episode.py`` builds an extra :class:`~isaaclab.managers.RewardManager` from
:func:`reference_reward_manager` and reports that manager's episodic returns next to the task ones.
Nothing here changes the reward the policy is trained or scored with.

Caveat: two reference terms are defined on the reference action space (joint torques, scale 7.5).
``action_l2`` penalizes the raw action and ``power_consumption`` multiplies it by the joint velocity.
This branch drives the joints with position targets instead, so those two terms read the normalized
action here; they are kept because the reference definition is what has to stay fixed.
"""

from __future__ import annotations

from isaaclab.managers import RewardManager, RewardTermCfg

import isaaclab.envs.mdp as core_mdp
import isaaclab_tasks.manager_based.classic.humanoid.mdp as humanoid_mdp

#: Commit whose Ant ``RewardsCfg`` this module copies (the tip of the cailab fork's ``main``).
REFERENCE_COMMIT = "e83a5d2f11ca1b5f03b690e1978479e620c500e2"


def reference_reward_terms() -> dict[str, RewardTermCfg]:
    """Reference Ant reward terms, keyed by the term names the reference version used."""
    return {
        "progress": RewardTermCfg(
            func=humanoid_mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)}
        ),
        "alive": RewardTermCfg(func=core_mdp.is_alive, weight=0.5),
        "upright": RewardTermCfg(func=humanoid_mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93}),
        "move_to_target": RewardTermCfg(
            func=humanoid_mdp.move_to_target_bonus,
            weight=0.5,
            params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)},
        ),
        "action_l2": RewardTermCfg(func=core_mdp.action_l2, weight=-0.005),
        "energy": RewardTermCfg(
            func=humanoid_mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}}
        ),
        "joint_pos_limits": RewardTermCfg(
            func=humanoid_mdp.joint_pos_limits_penalty_ratio,
            weight=-0.1,
            params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
        ),
    }


def reference_reward_manager(env) -> RewardManager:
    """Build a :class:`RewardManager` that scores ``env`` with the reference reward definition.

    The manager is independent of ``env.reward_manager``: it only feeds the evaluation report, so the
    environment keeps computing (and the policy keeps being scored with) its own reward terms.

    The manager's terms are stateful in the same way the environment's are (``progress_reward`` keeps
    the previous potential), so its episodes must be closed with ``manager.reset(env_ids)`` whenever
    the environment resets those environments.

    Args:
        env: The unwrapped environment the reference terms are evaluated against.

    Returns:
        The reference reward manager, with its term state seeded from the environment's current state
        (the environment has already been reset when the play script builds it).
    """
    manager = RewardManager(reference_reward_terms(), env)
    # zeroes the episodic sums and initializes progress_reward's potentials at the current pose
    manager.reset()
    return manager
