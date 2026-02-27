# Running PolaRiS Evaluations

## Steerable VLA Eval

### Basic usage

```bash
sbatch --export=ALL,EVAL_ARGS="--rollouts 1" eval_steerable_vla.sh
```

### Common overrides

```bash
# Change number of rollouts
sbatch --export=ALL,EVAL_ARGS="--rollouts 5" eval_steerable_vla.sh

# Change environment
sbatch --export=ALL,EVAL_ENV=DROID-FoodBussing,EVAL_ARGS="--rollouts 1" eval_steerable_vla.sh

# Change checkpoint
sbatch --export=ALL,SVLA_CHECKPOINT=/path/to/checkpoint.pt,EVAL_ARGS="--rollouts 1" eval_steerable_vla.sh

# Change unnorm key (dataset normalization stats)
sbatch --export=ALL,SVLA_UNNORM_KEY=bridge_orig,EVAL_ARGS="--rollouts 1" eval_steerable_vla.sh

# Change server port (if 8001 is in use)
sbatch --export=ALL,POLICY_PORT=8002,EVAL_ARGS="--rollouts 1" eval_steerable_vla.sh
```

### Interactive steering (mid-episode instruction updates)

Use `steer_eval.sh` — starts the server, waits for it, then runs the evaluator interactively.

```bash
# 1. Get an interactive node
srun --pty --account=weirdlab --partition=ckpt-all \
     --gres=gpu:a40:1 --cpus-per-task=16 --mem=64G bash

# 2. Run the steering script (from inside that shell)
bash steer_eval.sh [options]
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--steer-frequency N` | `50` | Prompt every N steps |
| `--rollouts N` | `1` | Number of episodes |
| `--environment ENV` | `DROID-FoodBussing` | IsaacLab env |
| `--port PORT` | `8001` | VLA server port |
| `--instruction TEXT` | _(from env file)_ | Starting instruction override |

Examples:
```bash
bash steer_eval.sh --steer-frequency 30 --rollouts 2
bash steer_eval.sh --steer-frequency 50 --instruction "pick up the apple"
bash steer_eval.sh --environment DROID-FoodBussing --port 8002 --steer-frequency 25
```

At each prompt:
```
  [checkpoint saved: runs/steer_.../episode_0_step_50.mp4]

[Step 50] Current instruction: 'pick up the block'
Enter new instruction (blank to keep):
```

Results go to `runs/steer_<timestamp>/`:

| File | Contents |
|------|----------|
| `episode_N.mp4` | Full episode video |
| `episode_N_step_S.mp4` | Checkpoint video up to step S |
| `steer_log.txt` | Per-prompt instruction history |
| `svla_trace.log` | Per-step VLA trace |
| `server.log` | VLA server log |

---

### Interactive steering (manual / legacy)

Use `--steer-frequency N` to pause every N steps and type a new language instruction.
Requires an interactive terminal (`srun --pty` or local session).

```bash
srun --pty --gres=gpu:a40:1 ... bash
# inside the interactive session, start the VLA server in the background, then:
uv run scripts/eval.py \
    --environment DROID-FoodBussing \
    --policy.client SteerableVLA \
    --policy.host 127.0.0.1 \
    --policy.port 8001 \
    --run-folder runs/test \
    --steer-frequency 50
```

Every 50 steps the terminal prints:
```
[Step 50] Current instruction: 'pick up the block'
Enter new instruction (blank to keep):
```
Type a new instruction and press Enter, or press Enter alone to keep the current one.

**SLURM batch jobs**: `--steer-frequency` is safe to include — when stdin is closed the prompt is
silently skipped and the current instruction continues unchanged.

### Results

Results are saved to `runs/eval_svla_<timestamp>/`:

| File | Contents |
|------|----------|
| `episode_N.mp4` | Side-by-side video (external cam \| wrist cam) |
| `eval_results.csv` | `episode, episode_length, success, progress` |
| `server.log` | VLA server stdout (model loading, Flask logs) |
| `svla_trace.log` | Per-step trace: action values, latency |

SLURM job logs go to `/mmfs1/gscratch/weirdlab/dg20/slurm_logs/eval_svla_<jobid>.{out,err}`.

### Defaults

| Variable | Default |
|----------|---------|
| `EVAL_ENV` | `DROID-FoodBussing` |
| `POLICY_PORT` | `8001` |
| `SVLA_UNNORM_KEY` | `bridge_orig` |
| `SVLA_CHECKPOINT` | `steerable-policies-maniskill/.cache/.../step-080000-epoch-09-loss=0.0506.pt` |
| `RUN_FOLDER` | `runs/eval_svla_<timestamp>` |

### Performance

- Model load + warmup: ~2-3 min (waits up to 30 min)
- Episode throughput: ~4.75 min/episode (450 steps @ ~1.67 it/s)
- VLA inference latency: ~290-300 ms/step

---

## π0.5 Eval

```bash
sbatch eval.sh
```

Results saved to `runs/eval_<timestamp>/`.

- Episode throughput: ~3 min/episode (450 steps @ ~3 it/s)

---

## Architecture Notes

### Steerable VLA pipeline

```
SLURM job (single node)
├── maniskill_vla.sif  →  serve_steerable_vla.py  (Flask, port 8001)
│     VLA: Prismatic (DINO+SigLIP + Llama-2-7b)
│     Input:  224×224 RGB image + language instruction
│     Output: 7D delta-EE action [dx,dy,dz,dRoll,dPitch,dYaw,gripper]
│
└── octi-lab-base.sif  →  scripts/eval.py  (Isaac Sim)
      Client: SteerableVLAClient
        - sends external cam image + instruction to Flask server
        - converts delta-EE → joint positions via DLS IK (pytorch_kinematics)
        - returns 8D action [j1..j7, gripper_binary]
```

### Key files

| File | Purpose |
|------|---------|
| `eval_steerable_vla.sh` | SLURM submission script |
| `scripts/serve_steerable_vla.py` | Flask VLA server |
| `src/polaris/policy/steerable_vla_client.py` | Isaac Sim client + IK |
| `src/polaris/policy/_ik_utils.py` | DLS IK utilities |
| `src/polaris/assets/panda.urdf` | Panda URDF for pytorch_kinematics |
