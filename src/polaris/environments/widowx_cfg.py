import torch
from pathlib import Path
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import isaaclab.envs.mdp as mdp
import numpy as np
from typing import Sequence

from polaris.environments.robot_cfg import WIDOWX

from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG

# Reuse the same camera pose fix from droid_cfg
from polaris.environments.droid_cfg import FixedCamera


### SceneCfg ###
@configclass
class SceneCfg(InteractiveSceneCfg):

    robot = WIDOWX

    wrist_cam = CameraCfg(
        class_type=FixedCamera,
        prim_path="{ENV_REGEX_NS}/robot/wx250s/wx250s_wrist_link/wrist_cam",
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/biglight",
        spawn=sim_utils.DomeLightCfg(intensity=1000),
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/wx250s/wx250s_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/wx250s/wx250s_gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

    def dynamic_setup(self, environment_path, robot_splat=True, nightmare="", **kwargs):
        environment_path_ = Path(environment_path)
        environment_path = str(environment_path_.resolve())

        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=environment_path,
                activate_contact_sensors=False,
            ),
        )
        self.scene = scene
        if not robot_splat:
            self.robot.spawn.semantic_tags = [("class", "raytraced")]
        stage = Usd.Stage.Open(environment_path)
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        for child in children:
            name = child.GetName()
            print(name)

            if child.IsA(UsdGeom.Camera):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    height=720,
                    width=1280,
                    data_types=["rgb", "semantic_segmentation"],
                    colorize_semantic_segmentation=False,
                    spawn=None,
                    offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
                )
                setattr(self, name, asset)
            elif UsdPhysics.RigidBodyAPI(child):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
                setattr(self, name, asset)

        if not hasattr(self, "external_cam"):
            self.external_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/scene/external_cam",
                height=720,
                width=1280,
                data_types=["rgb", "semantic_segmentation"],
                colorize_semantic_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.0476,
                    horizontal_aperture=2.5452,
                    vertical_aperture=1.4721,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(-0.01, -0.33, 0.48),
                    rot=(0.76, 0.43, -0.24, -0.42),
                    convention="opengl",
                ),
            )


### ActionCfg ###


class BinaryJointPositionVLAAction(BinaryJointPositionAction):
    """Gripper action matching VLA Bridge convention: 0=close, 1=open."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # VLA: >0.5 means open; base class uses >0 means open (opposite for [0,1] range)
        binary_mask = actions > 0.5  # True = open
        self._processed_actions = torch.where(
            binary_mask, self._open_command, self._close_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class BinaryJointPositionVLAActionCfg(BinaryJointPositionActionCfg):
    """Configuration for VLA-convention gripper: 0=close, 1=open."""

    class_type = BinaryJointPositionVLAAction

# WidowX arm joint names (6-DOF)
WIDOWX_ARM_JOINTS = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# WidowX gripper joint limits (prismatic, in metres):
#   left_finger:  [0.015, 0.037]  — 0.037 = open, 0.015 = closed
#   right_finger: [-0.037, -0.015] — mirrored direction in the USD
WIDOWX_GRIPPER_OPEN = 0.037
WIDOWX_GRIPPER_CLOSE = 0.015


@configclass
class ActionCfg:
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        body_name="wx250s_gripper_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=1.0,
    )

    left_finger = BinaryJointPositionVLAActionCfg(
        asset_name="robot",
        joint_names=["left_finger", "right_finger"],
        open_command_expr={
            "left_finger": WIDOWX_GRIPPER_OPEN,
            "right_finger": -WIDOWX_GRIPPER_OPEN,   # mirrored
        },
        close_command_expr={
            "left_finger": WIDOWX_GRIPPER_CLOSE,
            "right_finger": -WIDOWX_GRIPPER_CLOSE,  # mirrored
        },
    )


### ObsCfg ###
def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in WIDOWX_ARM_JOINTS
    ]
    return robot.data.joint_pos[:, joint_indices]


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name == "left_finger"
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    # rescale to [0, 1]: 0 = closed (0.015), 1 = open (0.037)
    joint_pos = (joint_pos - WIDOWX_GRIPPER_CLOSE) / (WIDOWX_GRIPPER_OPEN - WIDOWX_GRIPPER_CLOSE)
    return joint_pos


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class CommandsCfg:
    pass


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    pass


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.rerender_on_reset = True

    def dynamic_setup(self, *args):
        self.scene.dynamic_setup(*args)
