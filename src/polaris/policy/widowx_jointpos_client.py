"""
widowx_jointpos_client.py

PolaRiS InferenceClient for WidowX VLA policies using delta end-effector control.

The server returns a 7D action:
    [dx, dy, dz, dRoll, dPitch, dYaw, gripper∈[0,1]]

The 6D arm delta is passed directly to DifferentialInverseKinematicsActionCfg
in the Isaac Sim environment (no client-side IK conversion).
Gripper: 1=open, 0=close (Bridge convention).

Bridge-dataset image preprocessing (JPEG+lanczos) and observation history
are handled server-side; the client sends the raw camera image.

Registration key: "WidowXJointPos"
"""

import base64
import io
import logging
from typing import Optional

import cv2
import httpx
import numpy as np
from PIL import Image

from polaris.config import PolicyArgs
from polaris.policy.abstract_client import InferenceClient

logger = logging.getLogger(__name__)


def _to_numpy_uint8(img) -> np.ndarray:
    """Convert a tensor or array (optionally batched) to a numpy uint8 image."""
    if hasattr(img, "shape") and len(img.shape) == 4:
        img = img[0]
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    elif hasattr(img, "numpy"):
        img = img.numpy()
    return np.asarray(img, dtype=np.uint8)


def _encode_image_b64(img_rgb: np.ndarray) -> str:
    """Base64-encode an RGB uint8 image as JPEG (full resolution; server handles preprocessing)."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@InferenceClient.register(client_name="WidowXJointPos")
class WidowXJointPosClient(InferenceClient):
    """Single-step client for WidowX. Server output (7D delta EE) passed directly to DiffIK."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.unnorm_key: Optional[str] = getattr(args, "unnorm_key", None)

        self._base_url = f"http://{args.host}:{args.port}"
        self._http = httpx.Client(timeout=120.0, trust_env=False)

    @property
    def rerender(self) -> bool:
        return True  # single-step: always re-render

    def reset(self):
        """Clear server-side episode history."""
        try:
            self._http.post(f"{self._base_url}/reset")
        except Exception as exc:
            logger.warning("Failed to reset server episode history: %s", exc)

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Run one inference step.

        Returns:
            action: (7,) float32 – delta EE pose (6) + gripper binary (1)
            viz:    (224, 448, 3) uint8
        """
        ext_img = _to_numpy_uint8(obs["splat"]["external_cam"])
        wrist_img = _to_numpy_uint8(obs["splat"]["wrist_cam"])
        robot_state = obs["policy"]
        joint_pos: np.ndarray = (
            robot_state["arm_joint_pos"].clone().detach().cpu().float().numpy()[0]
        )  # (6,)

        # Server handles Bridge-dataset preprocessing (JPEG+lanczos) and observation history
        payload = {
            "image_b64": _encode_image_b64(ext_img),
            "instruction": instruction,
            "joint_positions": joint_pos.tolist(),
        }
        if self.unnorm_key is not None:
            payload["unnorm_key"] = self.unnorm_key

        resp = self._http.post(f"{self._base_url}/predict", json=payload)
        if resp.status_code >= 400:
            logger.error("Server returned %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        action7 = np.asarray(resp.json()["action"], dtype=np.float32)  # (7,)

        # action7[:6] = delta EE pose [dx, dy, dz, dRoll, dPitch, dYaw]
        # action7[6]  = gripper command ∈ [0, 1]; 1=open, 0=close (Bridge convention)
        gripper_binary = np.float32(1.0 if action7[6] > 0.5 else 0.0)
        action = np.append(action7[:6], gripper_binary).astype(np.float32)  # (7,)

        ext_small = cv2.resize(ext_img, (224, 224)).astype(np.uint8)
        wrist_small = cv2.resize(wrist_img, (224, 224)).astype(np.uint8)
        viz = np.concatenate([ext_small, wrist_small], axis=1)

        return action, viz

    def infer_batch(
        self, obs_list: list[dict], instruction: str,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Batch inference: single HTTP round-trip via /predict_batch."""
        if len(obs_list) == 1:
            return [self.infer(obs_list[0], instruction)]

        items = []
        images_for_viz = []  # (ext_img, wrist_img) per obs
        for obs in obs_list:
            ext_img = _to_numpy_uint8(obs["splat"]["external_cam"])
            wrist_img = _to_numpy_uint8(obs["splat"]["wrist_cam"])
            robot_state = obs["policy"]
            joint_pos = (
                robot_state["arm_joint_pos"].clone().detach().cpu().float().numpy()[0]
            )
            item = {
                "image_b64": _encode_image_b64(ext_img),
                "instruction": instruction,
                "joint_positions": joint_pos.tolist(),
            }
            if self.unnorm_key is not None:
                item["unnorm_key"] = self.unnorm_key
            items.append(item)
            images_for_viz.append((ext_img, wrist_img))

        resp = self._http.post(f"{self._base_url}/predict_batch", json={"items": items})
        if resp.status_code >= 400:
            logger.error("Server returned %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        raw_actions = resp.json()["actions"]

        results = []
        for action7_list, (ext_img, wrist_img) in zip(raw_actions, images_for_viz):
            action7 = np.asarray(action7_list, dtype=np.float32)
            gripper_binary = np.float32(1.0 if action7[6] > 0.5 else 0.0)
            action = np.append(action7[:6], gripper_binary).astype(np.float32)
            ext_small = cv2.resize(ext_img, (224, 224)).astype(np.uint8)
            wrist_small = cv2.resize(wrist_img, (224, 224)).astype(np.uint8)
            viz = np.concatenate([ext_small, wrist_small], axis=1)
            results.append((action, viz))
        return results
