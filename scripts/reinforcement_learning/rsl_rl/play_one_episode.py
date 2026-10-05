# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play one episode from a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play one episode with an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import subprocess
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


# =======code edit=======
# Episode reward report of the Ant tasks. The same episodes are scored with two reward definitions --
# the legacy (cailab) one and this branch's own -- and each gets its own labelled section, per-term
# table and result lines. The helpers below are used by main().


def _code_version() -> str:
    """Repository, branch and commit of the code that is running, e.g. ``owner/repo @ branch (abcdef)``."""
    try:
        root = os.path.dirname(os.path.abspath(__file__))

        def git(*args: str) -> str:
            return subprocess.check_output(("git", *args), cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

        url = git("config", "--get", "remote.origin.url")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        commit = git("rev-parse", "--short", "HEAD")
        # https://github.com/owner/repo.git and git@github.com:owner/repo.git both give owner/repo
        slug = url.removesuffix(".git").rstrip("/").split("://")[-1].replace(":", "/")
        return f"{'/'.join(slug.split('/')[-2:])} @ {branch} ({commit})"
    except Exception:
        # a copy of the repository without git metadata must not break the evaluation
        return "current working tree"


def _record_finished_episodes(env, manager, sums: torch.Tensor, lengths: torch.Tensor, recorded: torch.Tensor):
    """Wrap ``manager.reset`` so that the finished episodes are kept before their sums are cleared.

    Only the first episode of an environment is kept, in ``sums`` (per term, the manager's own episodic
    sums, i.e. term value x weight x dt) and ``lengths`` (episode length in steps).
    """
    reset = manager.reset

    def reset_with_recording(env_ids=None):
        ids = (
            torch.arange(sums.shape[0], device=sums.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=sums.device)
        )
        # episode_length_buf still holds the finished episode here and is zeroed only later in _reset_idx,
        # so a length of zero means this is not a finished episode (e.g. the reset done at construction).
        first = ids[(~recorded[ids]) & (env.episode_length_buf[ids] > 0)]
        if len(first) > 0:
            for column, name in enumerate(manager.active_terms):
                sums[first, column] = manager._episode_sums[name][first].to(dtype=torch.float64)
            lengths[first] = env.episode_length_buf[first].to(dtype=torch.float64)
            recorded[first] = True
        return reset(env_ids)

    manager.reset = reset_with_recording


def _episode_values(manager, sums: torch.Tensor, recorded: torch.Tensor) -> torch.Tensor:
    """Per environment and term: the kept first episode, or what the running episode has accumulated."""
    values = sums.clone()
    running = ~recorded
    if bool(running.any()):
        for column, name in enumerate(manager.active_terms):
            values[running, column] = manager._episode_sums[name][running].to(dtype=torch.float64)
    return values


def _print_reward_table(env, manager, sums: torch.Tensor, lengths: torch.Tensor, recorded: torch.Tensor) -> None:
    """Print one reward definition as a table: term, weight, episode return mean/std, per-step mean."""
    term_names = manager.active_terms
    values = _episode_values(manager, sums, recorded)
    steps = lengths.clone()
    running = ~recorded
    if bool(running.any()):
        steps[running] = env.episode_length_buf[running].to(dtype=torch.float64)
    steps = steps.clamp(min=1.0)

    weights = torch.tensor(
        [manager.get_term_cfg(name).weight for name in term_names], dtype=torch.float64, device=values.device
    )
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    # Divide the weight and the time step back out to get the reward function's own per-step output. The
    # denominator is negative for every penalty term, so it must keep its sign: clamping it to a small
    # positive number would print a garbage ~1e13 value instead. A zero weight becomes NaN.
    denominator = weights.unsqueeze(0) * env.step_dt * steps.unsqueeze(-1)
    denominator = torch.where(denominator.abs() < 1.0e-12, torch.full_like(denominator, float("nan")), denominator)
    step_mean = values / denominator
    total = values.sum(dim=1)

    print(f"[REWARD-STATS] First episode of {env.num_envs} environments (mean/std across environments):")
    print(f"[REWARD-STATS] {'reward term':<28}{'weight':>10}{'ep_return_mean':>16}{'ep_return_std':>15}{'step_mean':>13}")
    for column, name in enumerate(term_names):
        print(
            f"[REWARD-STATS] {name:<28}{weights[column].item():>10.4g}{mean[column].item():>16.6f}"
            f"{std[column].item():>15.6f} {step_mean[:, column].mean().item():>12.6f}"
        )
    print(
        f"[REWARD-STATS] {'TOTAL':<28}{'':>10}{total.mean().item():>16.6f}"
        f"{total.std(unbiased=False).item():>15.6f}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play one episode with an RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # configure the viewer to track the Ant root
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-4.0, 4.0, 2.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    episode_rewards = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # =======code edit=======
    # The Ant episode reward is reported twice, once with the reward functions and weights of the legacy
    # (cailab) version, commit e83a5d2f11ca1b5f03b690e1978479e620c500e2 (see ant/mdp/reference_reward.py),
    # and once with this branch's own retuned reward terms. An extra RewardManager evaluates the legacy
    # definition every step; the environment and the policy keep using the task's own reward.
    reference_manager = None
    legacy_term_sums = torch.zeros((env.num_envs, 0), dtype=torch.float64, device=env.device)
    legacy_lengths = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    legacy_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    task_term_sums = torch.zeros((env.num_envs, 0), dtype=torch.float64, device=env.device)
    task_lengths = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    task_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if "isaac-ant" in (args_cli.task or "").lower():
        from isaaclab_tasks.manager_based.classic.ant.mdp.reference_reward import (
            REFERENCE_COMMIT,
            REFERENCE_REF,
            REFERENCE_REPOSITORY,
            reference_reward_manager as build_reference_reward_manager,
        )

        task_manager = env.unwrapped.reward_manager
        reference_manager = build_reference_reward_manager(env.unwrapped)
        legacy_term_sums = torch.zeros(
            (env.num_envs, len(reference_manager.active_terms)), dtype=torch.float64, device=env.device
        )
        task_term_sums = torch.zeros(
            (env.num_envs, len(task_manager.active_terms)), dtype=torch.float64, device=env.device
        )
        _record_finished_episodes(env.unwrapped, task_manager, task_term_sums, task_lengths, task_recorded)
        _record_finished_episodes(env.unwrapped, reference_manager, legacy_term_sums, legacy_lengths, legacy_recorded)
        # Both definitions have to see the same episodes. The environment computes the task reward and
        # then resets the environments that finished, so the legacy reward is evaluated inside that same
        # window: hooking compute() scores the terminal step before the reset pose, and hooking reset()
        # (below) closes the legacy episode at the moment episode_length_buf still holds its length.
        task_reward_compute = task_manager.compute
        task_reward_reset = task_manager.reset

        def _compute_task_and_reference_reward(dt):
            task_reward = task_reward_compute(dt)
            reference_manager.compute(dt)
            return task_reward

        def _reset_task_and_reference_reward(env_ids=None):
            # the wrapped reset records both managers' finished episodes before they clear their sums
            task_reward = task_reward_reset(env_ids)
            reference_manager.reset(env_ids)
            return task_reward

        task_manager.compute = _compute_task_and_reference_reward
        task_manager.reset = _reset_task_and_reference_reward
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            # Include the terminal step, then ignore auto-reset episodes for finished environments.
            active = ~finished
            episode_rewards[active] += rewards[active]
            episode_steps[active] += 1
            finished |= dones.bool()
        timestep += 1

        # Wait for the first episode of every environment to finish.
        if finished.all().item():
            break

        # Recording length must not truncate episode statistics.
        if timestep >= env.max_episode_length:
            print(f"[INFO] Reached maximum episode length: {env.max_episode_length}")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    def _print_results(reward_total: torch.Tensor) -> None:
        """Result lines of one reward definition; the episode lengths do not depend on the definition."""
        completed = int(finished.sum().item())
        if completed == env.num_envs:
            print(f"[INFO] All {env.num_envs} environments finished their first episode.")
        print(f"[INFO] Completed first episodes: {completed}/{env.num_envs}")
        if completed != env.num_envs:
            print("[INFO] Statistics include partial episodes for unfinished environments.")
        if env.num_envs == 1:
            print(f"[RESULT] Episode reward total: {reward_total[0].item():.6f}")
            print(f"[RESULT] Episode steps: {episode_steps[0].item()}")
        else:
            # Population standard deviation across the evaluated environments.
            steps = episode_steps.to(dtype=torch.float64)
            print(
                f"[RESULT] Episode reward total: mean={reward_total.mean().item():.6f}, "
                f"std={reward_total.std(unbiased=False).item():.6f}"
            )
            print(
                f"[RESULT] Episode steps: mean={steps.mean().item():.6f}, "
                f"std={steps.std(unbiased=False).item():.6f}"
            )

    if reference_manager is None:
        # other tasks keep the single reward definition of their own environment configuration
        _print_results(episode_rewards)
    else:
        definitions = [
            (
                "LEGACY",
                f"{REFERENCE_REPOSITORY} @ {REFERENCE_REF} ({REFERENCE_COMMIT[:12]})",
                reference_manager,
                legacy_term_sums,
                legacy_lengths,
                legacy_recorded,
            ),
            (
                "OUR CODE",
                _code_version(),
                env.unwrapped.reward_manager,
                task_term_sums,
                task_lengths,
                task_recorded,
            ),
        ]
        # per-term tables first, then the result lines, both under the same section headers
        for kind, version, manager, sums, lengths, recorded in definitions:
            print(f"=== {kind} ({version}) ===")
            _print_reward_table(env.unwrapped, manager, sums, lengths, recorded)
        for kind, version, manager, sums, lengths, recorded in definitions:
            print(f"=== {kind} ({version}) ===")
            _print_results(_episode_values(manager, sums, recorded).sum(dim=1))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
