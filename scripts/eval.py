import sys
import tyro
import mediapy
import imageio_ffmpeg

# Use bundled ffmpeg binary (avoids dependency on system ffmpeg)
mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd
import numpy as np


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs


def main(eval_args: EvalArgs):
    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient

    num_envs = eval_args.num_envs
    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=num_envs,
        use_fabric=True,
    )
    env: ManagerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    rollouts = len(initial_conditions)
    if eval_args.instruction:
        language_instruction = eval_args.instruction
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    completed_episodes = len(episode_df)
    if completed_episodes >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    horizon = env.max_episode_length

    if num_envs == 1:
        # --- Single-env path (preserves original behavior) ---
        video = []
        bar = tqdm.tqdm(range(horizon))
        episode = completed_episodes
        obs, info = env.reset(
            object_positions=initial_conditions[episode % len(initial_conditions)]
        )
        policy_client.reset()
        print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
        steer_log_path = run_folder / "steer_log.txt"
        step = 0
        while True:
            # Steering prompt: pause every N steps to allow instruction update
            if eval_args.steer_frequency and step > 0 and step % eval_args.steer_frequency == 0:
                # Save checkpoint video so far
                if video:
                    ckpt_path = run_folder / f"episode_{episode}_step_{step}.mp4"
                    mediapy.write_video(ckpt_path, video, fps=15)

                bar.clear()  # pause tqdm so it doesn't overwrite the prompt
                if video:
                    print(f"  [checkpoint saved: {ckpt_path}]")
                print(f"\n[Step {step}] Current instruction: '{language_instruction}'")
                try:
                    import termios
                    # Read directly from /dev/tty so apptainer stdin redirection doesn't matter
                    with open("/dev/tty", "r") as tty:
                        termios.tcflush(tty, termios.TCIFLUSH)  # discard buffered keypresses
                        sys.stdout.write("Enter new instruction (blank to keep): ")
                        sys.stdout.flush()
                        new_instr = tty.readline().strip()
                    if new_instr:
                        language_instruction = new_instr
                        print(f"  → Updated to: '{language_instruction}'")
                    with open(steer_log_path, "a") as f:
                        f.write(f"episode={episode} step={step} instruction={language_instruction!r}\n")
                except (EOFError, OSError):
                    pass  # no terminal available (SLURM batch), continue silently
                bar.refresh()  # restore tqdm bar

            # obs["splat"] is always a list now; unwrap for single-env policy
            obs_for_policy = {**obs, "splat": obs["splat"][0]}
            action, viz = policy_client.infer(obs_for_policy, language_instruction)
            if viz is not None:
                video.append(viz)
            obs, rew, term, trunc, info = env.step(
                torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
            )

            step += 1
            bar.update(1)
            if term[0] or trunc[0] or bar.n >= horizon:
                policy_client.reset()
                step = 0

                # Save video and metadata
                filename = run_folder / f"episode_{episode}.mp4"
                mediapy.write_video(filename, video, fps=15)

                # Log episode results to CSV
                episode_data = {
                    "episode": episode,
                    "episode_length": bar.n,
                    "success": info["rubric"]["success"],
                    "progress": info["rubric"]["progress"],
                }
                episode_df = pd.concat(
                    [episode_df, pd.DataFrame([episode_data])], ignore_index=True
                )
                episode_df.to_csv(csv_path, index=False)

                bar.close()
                print(f"Episode {episode} finished. Episode length: {bar.n}")
                bar = tqdm.tqdm(range(horizon))

                episode += 1
                video = []
                if episode >= rollouts:
                    break

                obs, info = env.reset(
                    object_positions=initial_conditions[episode % len(initial_conditions)]
                )

    else:
        # --- Multi-env path ---
        start_episode = completed_episodes

        # Dispatch first num_envs episodes at init; clamp if fewer remain
        env_episode = list(range(start_episode, start_episode + num_envs))
        env_step = [0] * num_envs
        env_video = [[] for _ in range(num_envs)]
        env_active = [ep < rollouts for ep in env_episode]
        next_episode = start_episode + num_envs  # next episode index to assign

        # Initial reset with per-env initial conditions
        poses = [initial_conditions[ep % len(initial_conditions)] for ep in env_episode]
        obs, info = env.reset(object_positions=poses)
        policy_client.reset()

        remaining = rollouts - start_episode
        print(f" >>> Starting multi-env eval: {num_envs} envs, {remaining} episodes remaining <<< ")

        while any(env_active):
            # Collect actions for all active envs
            actions = [None] * num_envs
            for i in range(num_envs):
                if not env_active[i]:
                    continue
                obs_i = {"splat": obs["splat"][i]}
                if "policy" in obs:
                    obs_i["policy"] = {k: v[i:i+1] for k, v in obs["policy"].items()}
                action_i, viz_i = policy_client.infer(obs_i, language_instruction)
                actions[i] = action_i
                if viz_i is not None:
                    env_video[i].append(viz_i)

            # Fill inactive envs with zeros matching the active action shape
            first_action = next(a for a in actions if a is not None)
            for i in range(num_envs):
                if actions[i] is None:
                    actions[i] = np.zeros_like(first_action)

            actions_tensor = torch.tensor(np.stack(actions))
            obs, rew, term, trunc, info = env.step(
                actions_tensor, expensive=policy_client.rerender
            )

            for i in range(num_envs):
                if not env_active[i]:
                    continue
                env_step[i] += 1
                done = bool(term[i]) or bool(trunc[i]) or env_step[i] >= horizon
                if done:
                    ep_idx = env_episode[i]
                    episode_data = {
                        "episode": ep_idx,
                        "episode_length": env_step[i],
                        "success": info["rubric"]["success"],
                        "progress": info["rubric"]["progress"],
                    }
                    episode_df = pd.concat(
                        [episode_df, pd.DataFrame([episode_data])], ignore_index=True
                    )
                    episode_df.to_csv(csv_path, index=False)

                    if env_video[i]:
                        mediapy.write_video(
                            run_folder / f"episode_{ep_idx}.mp4", env_video[i], fps=15
                        )
                    print(f"Episode {ep_idx} finished. Episode length: {env_step[i]}")

                    if next_episode < rollouts:
                        env_episode[i] = next_episode
                        next_episode += 1
                        env.reset_single(
                            i, initial_conditions[env_episode[i] % len(initial_conditions)]
                        )
                        env_video[i] = []
                        env_step[i] = 0
                    else:
                        env_active[i] = False

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
