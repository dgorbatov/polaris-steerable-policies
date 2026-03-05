#!/usr/bin/env python3
"""
serve_steerable_vla.py

Flask-based inference server for the steerable OpenVLA (Prismatic) policy.
Uses Flask's built-in threaded WSGI server — no ASGI/uvicorn subprocess issues.

Must run inside the maniskill_vla.sif container (which has all prismatic deps).

Usage:
    python scripts/serve_steerable_vla.py \
        --checkpoint /path/to/step-XXXXXX-epoch-XX-loss=X.XXXX.pt \
        --repo /mmfs1/gscratch/weirdlab/dg20/steerable-policies-maniskill \
        --port 8001

Endpoints:
    POST /predict   – run one inference step
    POST /reset     – clear per-episode image history
    GET  /health    – {"status":"ok"} when model is ready
"""

import argparse
import base64
import os
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image

# ── Mutable state dict ────────────────────────────────────────────────────────
_state: dict = {
    "vla": None,
    "unnorm_key": "bridge_orig",
    "trace_log": None,
    "checkpoint": "",
    "replay_images": [],   # per-episode list of preprocessed (H,W,3) uint8 numpy arrays
    "obs_history": 1,
    "image_size": 224,
    "center_crop": True,
}


def _trace(msg: str) -> None:
    """Write msg to stdout AND a trace file (guarantees visibility)."""
    ts = time.strftime("%H:%M:%S")
    line = f"[SVLA {ts} PID={os.getpid()}] {msg}"
    print(line, flush=True)
    log_path = _state.get("trace_log")
    if log_path:
        try:
            with open(log_path, "a") as f:
                f.write(line + "\n")
                f.flush()
        except Exception:
            pass


_trace(f"module loaded")

# ── Flask app ─────────────────────────────────────────────────────────────────
from flask import Flask, jsonify, request as flask_request  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess_image(img_np: np.ndarray, resize_size: int) -> np.ndarray:
    """Replicate Bridge-dataset preprocessing: JPEG encode/decode + two-stage lanczos resize.

    Identical to maniskill_utils.get_maniskill_img (which mirrors simpler_utils.get_simpler_img).
    TF is available in maniskill_vla.sif.
    """
    import tensorflow as tf
    IMAGE_BASE_SIZE = 128
    image = tf.image.encode_jpeg(img_np)
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(
        image, (IMAGE_BASE_SIZE, IMAGE_BASE_SIZE), method="lanczos3", antialias=True
    )
    image = tf.image.resize(image, (resize_size, resize_size), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    loaded = _state["vla"] is not None
    return jsonify({"status": "ok", "model_loaded": loaded, "pid": os.getpid()})


@app.post("/reset")
def reset_episode():
    """Clear per-episode image history. Call at the start of each new episode."""
    _state["replay_images"].clear()
    _trace("Episode history cleared")
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    vla = _state["vla"]
    _trace(f"/predict called, vla={vla is not None}")
    if vla is None:
        return jsonify({"detail": "Model not loaded yet"}), 503

    t0 = time.perf_counter()
    data = flask_request.get_json(force=True)

    try:
        img_bytes = base64.b64decode(data["image_b64"])
        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil, dtype=np.uint8)
    except Exception as exc:
        return jsonify({"detail": f"Bad image: {exc}"}), 400

    instruction = data.get("instruction", "")
    unnorm_key = data.get("unnorm_key") or _state["unnorm_key"]

    # ── Preprocess: Bridge-dataset JPEG+lanczos pipeline (identical to training time) ──
    preprocessed = _preprocess_image(img_np, _state["image_size"])
    _state["replay_images"].append(preprocessed)

    # ── Build observation history (pad with first frame if needed) ──
    obs_history = _state["obs_history"]
    image_history = _state["replay_images"][-obs_history:]
    if len(image_history) < obs_history:
        pad = [_state["replay_images"][0]] * (obs_history - len(image_history))
        image_history = pad + image_history

    # ── Run VLA (PIL conversion + center crop + predict_action, identical to run_maniskill_eval.py) ──
    from robot.openvla_utils import get_prismatic_vla_action
    import torch
    observation = {"full_image": image_history}
    with torch.inference_mode():
        action = get_prismatic_vla_action(
            vla, None, _state["checkpoint"], observation, instruction, unnorm_key,
            center_crop=_state["center_crop"],
        )

    action_list = np.asarray(action, dtype=np.float32).flatten().tolist()  # get_prismatic_vla_action returns ndarray
    latency = (time.perf_counter() - t0) * 1000.0
    _trace(f"action={action_list}  latency={latency:.1f}ms")

    return jsonify({"action": action_list, "latency_ms": latency})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Steerable-VLA (Prismatic) Flask server")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt checkpoint file")
    parser.add_argument("--repo",
                        default="/mmfs1/gscratch/weirdlab/dg20/steerable-policies-maniskill",
                        help="Path to the steerable-policies-maniskill repo root")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace-log", default=None,
                        help="Path to a trace log file")
    parser.add_argument("--obs-history", type=int, default=1,
                        help="Observation history length (default: 1)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image resize target for preprocessing (default: 224)")
    parser.add_argument("--center-crop", dest="center_crop", action="store_true",
                        help="Apply center crop augmentation (default: on)")
    parser.add_argument("--no-center-crop", dest="center_crop", action="store_false",
                        help="Disable center crop augmentation")
    parser.set_defaults(center_crop=True)
    args = parser.parse_args()

    _state["unnorm_key"] = args.unnorm_key
    _state["trace_log"] = args.trace_log
    _state["checkpoint"] = args.checkpoint
    _state["obs_history"] = args.obs_history
    _state["image_size"] = args.image_size
    _state["center_crop"] = args.center_crop

    _trace(
        f"main() started, checkpoint={args.checkpoint}, port={args.port}, "
        f"obs_history={args.obs_history}, image_size={args.image_size}, "
        f"center_crop={args.center_crop}"
    )

    # Make prismatic/ importable
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)

    import torch
    from prismatic.models.load import load_vla

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        _trace("WARNING: HF_TOKEN not set — loading gated Llama-2 backbone may fail")

    _trace(f"Loading VLA checkpoint ...")
    vla = load_vla(args.checkpoint, hf_token=hf_token, load_for_training=False)
    _trace("load_vla() returned")

    _trace(f"Casting to half precision and moving to {args.device} ...")
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(args.device)
    vla.eval()
    _trace("Cast and move done")

    _trace("Running warmup inference ...")
    from robot.openvla_utils import get_prismatic_vla_action
    dummy_obs = {"full_image": [np.zeros((224, 224, 3), dtype=np.uint8)]}
    with torch.inference_mode():
        test_action = get_prismatic_vla_action(
            vla, None, args.checkpoint, dummy_obs, "test", args.unnorm_key,
            center_crop=args.center_crop,
        )
    _trace(f"Warmup done. Action shape: {np.asarray(test_action).shape}")

    # Warm up TF preprocessing — `import tensorflow as tf` is deferred inside
    # _preprocess_image(); triggering it here prevents a >30s stall on the first
    # real /predict request after Flask starts.
    _trace("Warming up TF image preprocessing ...")
    _preprocess_image(np.zeros((224, 224, 3), dtype=np.uint8), args.image_size)
    _trace("TF preprocessing warmup done")

    _state["vla"] = vla
    _trace(f"VLA stored in _state. vla is not None: {_state['vla'] is not None}")

    _trace(f"Starting Flask server on {args.host}:{args.port} ...")
    # threaded=True: each request in its own thread, all sharing this process memory
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
