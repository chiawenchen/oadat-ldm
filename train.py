## train.py: simple script for training a diffusion model using PyTorch Lightning.

import os
import torch
from config import TrainingConfig, parse_arguments
from datamodule import OADATDataModule
from model import DiffusionModel
from utils import get_last_checkpoint, transforms
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from diffusers import DDIMScheduler


# the precision settings for matrix multiplications
torch.set_float32_matmul_precision("medium")


def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()

    # Set up training configuration
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    # Set up data module
    datamodule = OADATDataModule(
        data_path=args.oadat_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_swfd_scd=args.mix_swfd_scd,
        transforms=transforms
    )

    # Set up noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=1e-5,
        beta_end=5e-3,
        beta_schedule="scaled_linear",
    )

    # Set up sample image saving path
    sample_dir = os.path.join(config.output_dir, "samples", args.job_name)
    os.makedirs(sample_dir, exist_ok=True)
    config.sample_dir = sample_dir

    # Paths for storing fixed noisy images
    fixed_image_paths = {
        "val": os.path.join(config.sample_dir, "val_fixed_noisy_images"),
        "scd": os.path.join(config.sample_dir, "scd_fixed_noisy_images"),
    }
    os.makedirs(fixed_image_paths["val"], exist_ok=True)
    os.makedirs(fixed_image_paths["scd"], exist_ok=True)
    config.fixed_image_paths = fixed_image_paths

    # Initialize the model
    model = DiffusionModel(config=config, noise_scheduler=noise_scheduler)

    # Set up logger
    logger = WandbLogger(
        project="oadat-ldm",
        name=args.job_name,
        log_model=False,
        config=config.__dict__,
    )

    # Set up checkpoint callback
    ckpt_dir = os.path.join(config.output_dir, "checkpoints", args.job_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        save_last=True,
        filename="{epoch:02d}-{val_loss:.4f}",
    )

    # Set up Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)], # depth=-1 for full summary
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )

    # Load the latest checkpoint if available
    latest_ckpt = get_last_checkpoint(ckpt_dir)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
    else:
        print("No checkpoint found, starting from scratch.")

    # Train the model
    trainer.fit(model, datamodule, ckpt_path=latest_ckpt)

    # print("best model path: ", checkpoint_callback.best_model_path, "best model score: ", checkpoint_callback.best_model_score)
    # trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
