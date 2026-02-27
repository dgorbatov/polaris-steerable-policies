"""
steerable_vla_client.py

PolaRiS InferenceClient for a steerable OpenVLA policy served via FastAPI.

The server (scripts/serve_steerable_vla.py) returns 7D delta-EE actions:
    [dx, dy, dz, dRoll, dPitch, dYaw, gripper]

This client:
  1. Encodes the external camera image and sends it with the instruction
  2. Receives the 7D delta-EE + gripper action from the server
  3. Converts delta EE → absolute joint positions via DLS IK (_ik_utils)
  4. Returns an 8D action [j1..j7, gripper_binary]

Registration key: "SteerableVLA"

Example PolicyArgs:
    PolicyArgs(
        client="SteerableVLA",
        host="127.0.0.1",
        port=8001,
        open_loop_horizon=None,   # single-step, no chunking
    )
    # Optionally set on the args object before use:
    args.unnorm_key = "bridge_orig"   # dataset unnorm key for OpenVLA
    args.ik_device  = "cuda"          # torch device for IK solver
    args.image_size = 224             # resize target sent to server
"""

import base64
import io
import logging
from functools import lru_cache
from typing import Optional

import cv2
import httpx
import numpy as np
from PIL import Image

from polaris.config import PolicyArgs
from polaris.policy.abstract_client import InferenceClient

logger = logging.getLogger(__name__)


# ── Lazy IK chain loader ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_chain(device: str):
    """Load and cache the pytorch_kinematics serial chain for panda_hand."""
    import pytorch_kinematics as pk  # type: ignore
    from polaris.policy._ik_utils import _find_panda_urdf

    urdf_path = _find_panda_urdf()
    with open(urdf_path) as f:
        urdf_str = f.read()
    chain = pk.build_serial_chain_from_urdf(urdf_str, "panda_hand").to(device=device)
    logger.info("IK chain built from %s on %s", urdf_path, device)
    return chain


# ── Image helpers ─────────────────────────────────────────────────────────────

def _encode_image_b64(img_rgb: np.ndarray, size: int = 224) -> str:
    """Resize an RGB uint8 image to size×size and base64-encode as JPEG."""
    resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    pil_img = Image.fromarray(resized.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Client ────────────────────────────────────────────────────────────────────

@InferenceClient.register(client_name="SteerableVLA")
class SteerableVLAClient(InferenceClient):
    """Single-step steerable-VLA client with IK conversion."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.unnorm_key: Optional[str] = getattr(args, "unnorm_key", None)
        self.ik_device: str = getattr(args, "ik_device", "cuda")
        self.image_size: int = getattr(args, "image_size", 224)

        self._base_url = f"http://{args.host}:{args.port}"
        # trust_env=False: disable proxy env-var reading (cluster uses Squid which
        # intercepts HTTP and returns 503 for localhost targets)
        self._http = httpx.Client(timeout=30.0, trust_env=False)

        # Eagerly load IK chain so first inference is not slow
        try:
            _get_chain(self.ik_device)
        except Exception as exc:
            logger.warning("IK chain not pre-loaded (will retry at first infer): %s", exc)

    # ── InferenceClient interface ──────────────────────────────────────────

    @property
    def rerender(self) -> bool:
        return True  # single-step: always re-render

    def reset(self):
        pass  # stateless client

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Run one inference step.

        Args:
            obs:         observation dict from PolaRiS env
            instruction: language instruction string
            return_viz:  if True, also return side-by-side camera view

        Returns:
            action: (8,) float32 – absolute joint positions (7) + gripper binary (1)
            viz:    (224, 448, 3) uint8 or None
        """
        ext_img: np.ndarray = obs["splat"]["external_cam"]   # H×W×3 uint8
        wrist_img: np.ndarray = obs["splat"]["wrist_cam"]    # H×W×3 uint8
        robot_state = obs["policy"]
        joint_pos: np.ndarray = (
            robot_state["arm_joint_pos"].clone().detach().cpu().float().numpy()[0]
        )  # (7,)

        # Build and send request
        payload = {
            "image_b64": _encode_image_b64(ext_img, self.image_size),
            "instruction": instruction,
            "joint_positions": joint_pos.tolist(),
        }
        if self.unnorm_key is not None:
            payload["unnorm_key"] = self.unnorm_key

        resp = self._http.post(f"{self._base_url}/predict", json=payload)
        if resp.status_code >= 400:
            logger.error("Server returned %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        delta7 = np.asarray(resp.json()["action"], dtype=np.float32)  # (7,)

        # delta7[:6] = [dx, dy, dz, dRoll, dPitch, dYaw]
        # delta7[6]  = gripper command ∈ [0, 1]
        chain = _get_chain(self.ik_device)
        from polaris.policy._ik_utils import _delta_ee_to_joint_pos

        new_joint_pos = _delta_ee_to_joint_pos(
            chain,
            joint_pos,
            delta7[:6],
            device=self.ik_device,
        )  # (7,) float32

        gripper_binary = np.float32(1.0 if delta7[6] > 0.5 else 0.0)
        action = np.append(new_joint_pos, gripper_binary).astype(np.float32)  # (8,)

        # Always compute viz so eval.py's video writer gets frames every step
        ext_small = cv2.resize(ext_img, (224, 224)).astype(np.uint8)
        wrist_small = cv2.resize(wrist_img, (224, 224)).astype(np.uint8)
        viz = np.concatenate([ext_small, wrist_small], axis=1)

        return action, viz
