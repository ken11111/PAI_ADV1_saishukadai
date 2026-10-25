# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Tatsuya Kamijo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tatsuya Kamijo (based on HuggingFace LeRobot lerobot/examples/3_train_policy.py)

"""
[pick180 用] 教材の scripts/train.py を元に、ACT のモデルの大きさを train_config.yaml から指定できるようにしたもの。
変更点: dim_model / n_heads / dim_feedforward / n_encoder_layers / n_decoder_layers を設定可能にし、
学習の途中経過として 1000 ステップごとにモデルを保存する。
環境変数 TRAIN_SEED で乱数の seed(初期値・データの順番)を固定し、TRAIN_OUTPUT で保存先を変えられる。

Train ACT (Action Chunking with Transformer) policy on the collected dataset.

This script trains an ACT policy using configuration parameters specified in:
    - configs/train_config.yaml: Training hyperparameters like batch size, learning rate, etc.
    - configs/crane_features.py: Dataset and feature configuration

Usage:
    pixi run train

"""


from pathlib import Path
import importlib.util
import os
import random
import numpy as np
import wandb
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

def load_config_py(config_path: Path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def load_train_config(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# =====================
# Load parameters from YAML
# =====================
train_cfg = load_train_config("configs/train_config.yaml")
DATASET_ROOT = Path(train_cfg["dataset_root"])
REPO_ID = train_cfg["repo_id"]
OUTPUT_DIRECTORY = Path(os.environ.get("TRAIN_OUTPUT") or train_cfg["output_directory"])
SEED = os.environ.get("TRAIN_SEED")
CONFIG_PATH = Path(train_cfg["config_path"])
DEVICE = torch.device(train_cfg["device"])
TRAINING_STEPS = train_cfg["training_steps"]
LOG_FREQ = train_cfg["log_freq"]
NUM_WORKERS = train_cfg["num_workers"]
BATCH_SIZE = train_cfg["batch_size"]
CHUNK_SIZE = train_cfg["chunk_size"]
N_ACTION_STEPS = train_cfg["n_action_steps"]
LEARNING_RATE = train_cfg["learning_rate"]
DELTA_TIMESTAMPS = train_cfg["delta_timestamps"]
MODEL_KW = {k: train_cfg[k] for k in ("dim_model", "n_heads", "dim_feedforward", "n_encoder_layers", "n_decoder_layers") if k in train_cfg}
# =====================

def main():
    if SEED is not None:
        random.seed(int(SEED)); np.random.seed(int(SEED)); torch.manual_seed(int(SEED))
        print(f"seed: {SEED}")
    # Create a directory to store the training checkpoint.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    config = load_config_py(CONFIG_PATH)
    features = config.features

    wandb.init(project="pai_crane_gazebo", config={
        "repo_id": REPO_ID,
        "dataset_root": DATASET_ROOT,
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "device": str(DEVICE),
    })

    dataset_metadata = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features: {input_features}")
    print(f"output_features: {output_features}")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        **MODEL_KW,
    )
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    print(f"model params: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M  ({MODEL_KW})")
    policy.train()
    policy.to(DEVICE)

    # Instantiate the dataset
    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, delta_timestamps=DELTA_TIMESTAMPS)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(SEED)) if SEED is not None else None,
    )

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # Remove the sequence_length dimension to adapt to the ACT model implementation
            batch["observation.state"] = batch["observation.state"][:, -1, :]  # (B, D)
            batch["observation.environment_state"] = batch["observation.environment_state"][:, -1, :]  # (B, D)
            # action is left as is
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
                wandb.log({"loss": loss.item(), "step": step})
            step += 1
            if step % 1000 == 0 and step < TRAINING_STEPS:
                policy.save_pretrained(OUTPUT_DIRECTORY / f"step_{step:06d}")
            if step >= TRAINING_STEPS:
                done = True
                break

    policy.save_pretrained(OUTPUT_DIRECTORY)
    print(f"policy saved to {OUTPUT_DIRECTORY}")
    wandb.finish()

if __name__ == "__main__":
    main()
