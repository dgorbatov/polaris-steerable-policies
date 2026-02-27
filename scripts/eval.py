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
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env: MangerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

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
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
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

        action, viz = policy_client.infer(obs, language_instruction)
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
            obs, info = env.reset(
                object_positions=initial_conditions[episode % len(initial_conditions)]
            )

            episode += 1
            video = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
