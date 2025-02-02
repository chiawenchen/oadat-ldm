# config.py: default config for training

import argparse
from dataclasses import dataclass, field
import torch

import ast

def parse_numbers(numbers_str):
    """
    Parse a string like '[1, 2, 3, 4]' into a Python list of numbers.
    """
    try:
        # Safely evaluate the string to a Python list
        numbers = ast.literal_eval(numbers_str)
        if isinstance(numbers, list):
            return numbers
        else:
            raise ValueError
    except Exception:
        raise argparse.ArgumentTypeError("Invalid format for --numbers. Use: [1, 2, 3, 4]")


@dataclass
class TrainingConfig:
    # default hyperparameters
    image_size: int = 256
    batch_size: int = 32
    num_epochs: int = 50
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

@dataclass
class LDMTrainingConfig:
    # default hyperparameters
    image_size: int = 256
    latent_channels: int = 3
    latent_size: int = 32
    num_down_blocks: int = 4
    num_up_blocks: int = 4
    num_classes: int = 2
    block_out_channels: list = field(default_factory=lambda: [64, 64, 128, 256])
    kl_loss_weight: float = 1.0e-06
    disc_start: int = 5551
    disc_weight: float = 0.5
    batch_size: int = 128
    num_epochs: int = 250
    learning_rate: float = 1e-4
    lr_warmup_epochs: int = 5
    save_image_epochs: int = 1
    seed: int = 42
    num_train_timesteps: int = 1000
    sample_dir: str = None
    sample_num: int = 11
    output_dir: str = "/mydata/dlbirhoui/chia/"  # directory to save models and images
    vae_ckpt_dir: str = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2/epoch=149-val_total_loss=0.0000.ckpt"
    cvae_ckpt_dir: str = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_lpips_disc_clf_cvae/last.ckpt"
    adv_weight: float = 5000
    classifier_weight: float = 1000.0
    classifier_scale: float = -1.0

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
        "--condition_ldm",
        action="store_true",
        help="Condition Latent Diffusion Model on class labels (set to True if specified, False otherwise)",
    )
    parser.add_argument(
        "--condition_vae",
        action="store_true",
        help="Condition VAE on class labels (set to True if specified, False otherwise)",
    )
    parser.add_argument(
        "--noise_schedule", default="linear", type=str,
        help="Noise Scheduler type",
    )
    parser.add_argument(
        "--classifier", default="resnet", type=str,
        help="Classifier type",
    )
    parser.add_argument(
        "--block_out_channels",
        type=parse_numbers,
        help="block_out_channels",
    )
    return parser.parse_args()
