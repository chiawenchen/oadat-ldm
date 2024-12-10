# config.py: default config for training

import argparse
from dataclasses import dataclass, field
import torch

@dataclass
class TrainingConfig:
    # default hyperparameters
    image_size: int = 256
    batch_size: int = 32
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_epochs: int = 5
    save_image_epochs: int = 5
    mixed_precision: str = (
        "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    )
    output_dir: str = "/mydata/dlbirhoui/chia/"  # directory to save models and images
    seed: int = 42
    num_train_timesteps: int = 1000
    sample_dir: str = None
    sample_num: int = 11

@dataclass
class ClassifierConfig:
    im_channels: int = 1
    down_ch: list = field(default_factory=lambda: [32, 64, 128, 256])
    mid_ch: list = field(default_factory=lambda: [256, 256, 128])
    down_sample: list = field(default_factory=lambda: [True, True, True])
    use_self_attention: list = field(default_factory=lambda: [False, False, True])
    t_emb_dim: int = 128
    num_downc_layers: int = 2
    num_midc_layers: int = 2
    use_scale_shift_norm: bool = True
    num_classes: int = 2
    learning_rate: float = 1e-4
    num_train_timesteps: int = 1000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on OADAT data."
    )
    parser.add_argument(
        "--oadat_dir", default="oadat", type=str, help="Path to the OADAT data folder"
    )
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="learning_rate"
    )
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of dataloader workers"
    )
    parser.add_argument("--job_name", default="ddim-test", type=str, help="Job name")
    parser.add_argument(
        "--mix_swfd_scd",
        action="store_true",
        help="Mix SWFD and SCD datasets (set to True if specified, False otherwise)",
    )
    parser.add_argument(
        "--noise_schedule", default="linear", type=str,
        help="Noise Scheduler type",
    )
    parser.add_argument(
        "--classifier", default="resnet", type=str,
        help="Classifier type",
    )
    # parser.add_argument(
    #     "--balance_class",
    #     type=bool,
    #     default=False,
    #     help="balance classes for classifier training (True or False)",
    # )
    return parser.parse_args()
