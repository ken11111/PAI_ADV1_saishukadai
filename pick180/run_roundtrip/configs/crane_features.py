# [roundtrip 用] observation.environment_state を、ブロックの位置(/cube_pose の x, y, z)と、動き出してからの経過時間(秒)にしたもの
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["cube_x", "cube_y", "cube_z", "time_s"],
    },
    "action": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
}
# Below is used for the dataset conversion
robot_type = "crane_plus"
repo_id = "crane_plus_roundtrip"
root = "/home/ubuntu/dataset/lerobot_dataset/crane_plus_roundtrip"
task_name = "pick_rotate_180_and_return"
