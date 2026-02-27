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
_state: dict = {"vla": None, "unnorm_key": "bridge_orig", "trace_log": None}


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


@app.get("/health")
def health():
    loaded = _state["vla"] is not None
    return jsonify({"status": "ok", "model_loaded": loaded, "pid": os.getpid()})


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
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"detail": f"Bad image: {exc}"}), 400

    instruction = data.get("instruction", "")
    unnorm_key = data.get("unnorm_key") or _state["unnorm_key"]

    import torch
    with torch.inference_mode():
        action, _ = vla.predict_action(image, instruction, unnorm_key=unnorm_key)

    action_list = np.asarray(action, dtype=np.float32).flatten().tolist()
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
    args = parser.parse_args()

    _state["unnorm_key"] = args.unnorm_key
    _state["trace_log"] = args.trace_log

    _trace(f"main() started, checkpoint={args.checkpoint}, port={args.port}")

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
    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    with torch.inference_mode():
        test_action, _ = vla.predict_action(dummy_img, "test", unnorm_key=args.unnorm_key)
    _trace(f"Warmup done. Action shape: {np.asarray(test_action).shape}")

    _state["vla"] = vla
    _trace(f"VLA stored in _state. vla is not None: {_state['vla'] is not None}")

    _trace(f"Starting Flask server on {args.host}:{args.port} ...")
    # threaded=True: each request in its own thread, all sharing this process memory
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
