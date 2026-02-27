"""
_ik_utils.py

Shared IK utilities for converting 7D delta EE actions (dx, dy, dz, dR, dP, dY, gripper)
to 8D absolute joint positions (j1..j7, gripper_binary) for the Franka Panda robot.
Uses pytorch_kinematics for FK + damped-least-squares IK.
"""

from pathlib import Path
from typing import Optional

import numpy as np

# Franka Panda joint limits [lower, upper] in radians
PANDA_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

# Scale constants for delta EE → workspace units
# These may need tuning against dataset_statistics.json q01/q99 ranges
_SCALE_POS = 1.0   # delta position scale (metres)
_SCALE_ROT = 1.0   # delta rotation scale (radians)


def _find_panda_urdf() -> str:
    """
    Locate the Franka Panda URDF file.

    Search order:
      1. Local assets: src/polaris/assets/panda.urdf  (relative to this file)
      2. IsaacLab install (inside container)
    """
    # 1. Local copy
    local = Path(__file__).parent.parent / "assets" / "panda.urdf"
    if local.exists():
        return str(local)

    # 2. IsaacLab bundled assets
    try:
        import isaaclab  # type: ignore
        matches = list(Path(isaaclab.__file__).parent.rglob("panda.urdf"))
        if matches:
            return str(matches[0])
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find panda.urdf. "
        "Place it at src/polaris/assets/panda.urdf or install isaaclab."
    )


def _delta_ee_to_joint_pos(
    chain,
    joint_pos: np.ndarray,
    delta_ee: np.ndarray,
    device: str = "cuda",
    damping: float = 0.05,
) -> np.ndarray:
    """
    Convert 6D delta EE pose to absolute joint positions via DLS IK.

    Args:
        chain:      pytorch_kinematics serial chain for panda_hand
        joint_pos:  current joint angles, shape (7,) float
        delta_ee:   6D delta [dx, dy, dz, dR, dP, dY] in EE frame
        device:     torch device string
        damping:    damping coefficient for damped-least-squares IK

    Returns:
        new joint positions, shape (7,) float32, clamped to Panda limits
    """
    import torch
    import pytorch_kinematics as pk  # type: ignore

    q = torch.tensor(joint_pos, dtype=torch.float32, device=device).unsqueeze(0)  # (1,7)

    # FK: get current EE transform
    fk_result = chain.forward_kinematics(q)
    # fk_result is a Transform3d; get the 4x4 matrix
    T_cur = fk_result.get_matrix()  # (1,4,4)

    # Build target transform by applying delta
    dx, dy, dz = delta_ee[0] * _SCALE_POS, delta_ee[1] * _SCALE_POS, delta_ee[2] * _SCALE_POS
    dR, dP, dY = delta_ee[3] * _SCALE_ROT, delta_ee[4] * _SCALE_ROT, delta_ee[5] * _SCALE_ROT

    # Small-angle rotation matrix from delta RPY (intrinsic XYZ)
    cR, sR = np.cos(dR), np.sin(dR)
    cP, sP = np.cos(dP), np.sin(dP)
    cY, sY = np.cos(dY), np.sin(dY)
    Rx = np.array([[1, 0, 0], [0, cR, -sR], [0, sR, cR]])
    Ry = np.array([[cP, 0, sP], [0, 1, 0], [-sP, 0, cP]])
    Rz = np.array([[cY, -sY, 0], [sY, cY, 0], [0, 0, 1]])
    dRot = (Rz @ Ry @ Rx).astype(np.float32)

    # Build 4x4 delta transform
    dT = np.eye(4, dtype=np.float32)
    dT[:3, :3] = dRot
    dT[:3, 3] = [dx, dy, dz]
    dT_t = torch.tensor(dT, device=device).unsqueeze(0)  # (1,4,4)

    # Target = current * delta (delta expressed in EE frame)
    T_target = torch.bmm(T_cur, dT_t)  # (1,4,4)

    # Jacobian-based DLS IK
    J = chain.jacobian(q)  # (1,6,7)
    J = J.squeeze(0)       # (6,7)

    # Current vs target EE position error
    pos_err = T_target[0, :3, 3] - T_cur[0, :3, 3]  # (3,)

    # Rotation error via log map (skew-symmetric part of R_target @ R_cur^T)
    R_cur = T_cur[0, :3, :3]
    R_tgt = T_target[0, :3, :3]
    R_err = R_tgt @ R_cur.T
    # vex of skew-symmetric part
    rot_err = torch.stack([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) * 0.5  # (3,)

    task_err = torch.cat([pos_err, rot_err], dim=0).unsqueeze(1)  # (6,1)

    # DLS: dq = J^T (J J^T + λ²I)^{-1} err
    lam2 = damping ** 2
    A = J @ J.T + lam2 * torch.eye(6, device=device)  # (6,6)
    dq = (J.T @ torch.linalg.solve(A, task_err)).squeeze(1)  # (7,)

    q_new = (q.squeeze(0) + dq).cpu().numpy()

    # Clamp to joint limits
    q_new = np.clip(q_new, PANDA_LIMITS[:, 0], PANDA_LIMITS[:, 1])

    return q_new.astype(np.float32)
