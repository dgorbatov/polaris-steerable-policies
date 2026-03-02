import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from polaris.utils import DATA_PATH

NVIDIA_DROID = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DATA_PATH / "nvidia_droid/noninstanceable.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": 0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=None,
            damping=None,
            effort_limit=200.0,
            velocity_limit=5.0,  # 2.175,
        ),
    },
)

# WidowX 250s (wx250s) - Interbotix 6-DOF arm
# Joint names are standard Interbotix URDF names — verify against your USD if they differ.
WIDOWX = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DATA_PATH / "widowx/wx250s_processed.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "waist": 0.0,
            "shoulder": -np.pi / 9,
            "elbow": np.pi / 9,
            "forearm_roll": 0.0,
            "wrist_angle": 0.0,
            "wrist_rotate": 0.0,
            "left_finger": 0.037,    # open  (range [0.015, 0.037])
            "right_finger": -0.037,  # open  (range [-0.037, -0.015], mirrored)
            "gripper": 0.0,          # passive joint (exists in USD, not in URDF)
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "wx_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["waist", "shoulder", "elbow", "forearm_roll"],
            effort_limit=3.8,
            velocity_limit=5.24,
            stiffness=100.0,
            damping=20.0,
        ),
        "wx_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_angle", "wrist_rotate"],
            effort_limit=1.5,
            velocity_limit=6.60,
            stiffness=100.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            stiffness=None,
            damping=None,
            effort_limit=50.0,
            velocity_limit=5.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            stiffness=None,
            damping=None,
            effort_limit=50.0,
            velocity_limit=5.0,
        ),
    },
)
