import torch
import argparse
from dataclasses import dataclass, field

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on OADAT data."
    )
    parser.add_argument(
        "--config_path", default="/mydata/dlbirhoui/chia/oadat-ldm/config/diffusion-model.yaml", type=str, help="Path to the config"
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
    return parser.parse_args()