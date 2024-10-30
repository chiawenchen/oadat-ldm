# config.py: default config for training

import argparse
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # default hyperparameters
    image_size: int = 256
    batch_size: int = 32
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_epochs: int = 5
    save_image_epochs: int = 1
    mixed_precision: str = (
        "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    )
    output_dir: str = "/mydata/dlbirhoui/chia/"  # directory to save models and images
    seed: int = 42
    num_train_timesteps: int = 1000
    sample_dir: str = None
    sample_num: int = 11
    fixed_image_paths: dict[str, str] = None


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
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of dataloader workers"
    )
    parser.add_argument("--job_name", default="ddim-test", type=str, help="Job name")
    parser.add_argument(
        "--mix_swfd_scd",
        type=bool,
        default=False,
        help="Mix SWFD and SCD datasets (True or False)",
    )
    return parser.parse_args()
