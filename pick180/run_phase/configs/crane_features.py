# [roundtrip 工程つき] observation.environment_state を、ブロックの位置(x, y, z)、工程の番号(ブロックを放した回数)、その工程に入ってからの経過時間(秒)にしたもの
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["cube_x", "cube_y", "cube_z", "phase", "phase_time_s"],
    },
    "action": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
}
# Below is used for the dataset conversion
robot_type = "crane_plus"
repo_id = "crane_plus_roundtrip_phase"
root = "/home/ubuntu/dataset/lerobot_dataset/crane_plus_roundtrip_phase"
task_name = "pick_rotate_180_and_return"
