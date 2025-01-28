import os
import numpy as np
import h5py
import wandb
from PIL import Image
import random

from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2

from diffusers import DDIMScheduler
from config.config import LDMTrainingConfig, parse_arguments
from utils import get_last_checkpoint, get_named_beta_schedule
import dataset
from datamodule import OADATDataModule

# from models.VAE import VAE
from models.AutoencoderKL import VAE
from models.AutoencoderKL_clf2_sigmoid import VAE as condition_VAE

torch.set_float32_matmul_precision("medium")


def main():

    # Parse command-line arguments
    args = parse_arguments()

    config = LDMTrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        # block_out_channels=args.block_out_channels
    )

    # sample directory
    sample_dir = os.path.join("/mydata/dlbirhoui/chia/samples/vae", args.job_name)
    os.makedirs(sample_dir, exist_ok=True)
    config.sample_dir = sample_dir

    # vae model
    if args.condition_vae:
        model = condition_VAE(config=config)
    else:
        model = VAE(config=config)

    # DataModule
    indices_scd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_train_indices.npy")
    indices_swfd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/train_sc_BP_indices.npy")
    datamodule = OADATDataModule(
        data_path=args.oadat_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_swfd_scd=args.mix_swfd_scd,
        indices_swfd=indices_swfd,
        indices_scd=indices_scd,
        return_labels=args.condition_vae
    )

    # Set up checkpoint callback
    ckpt_dir = os.path.join("/mydata/dlbirhoui/chia/checkpoints/vae", args.job_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    config.vae_ckpt_dir = ckpt_dir
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=3,
        monitor="val/total_loss",
        mode="min",
        save_weights_only=False,
        save_last=True,
        filename="{epoch:02d}-{val_total_loss:.4f}",
    )

    # Set up logger
    logger = WandbLogger(
        project="vae",
        name=args.job_name,
        log_model=False,
        config=config.__dict__
    )
    
    # Trainer with Wandb logger
    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        # precision='16-mixed',
    )

    # Load the latest checkpoint if available
    latest_ckpt = get_last_checkpoint(ckpt_dir)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
    else:
        print("No checkpoint found, starting from scratch.")

    # Fit model
    trainer.fit(model, datamodule, ckpt_path=latest_ckpt)


if __name__ == "__main__":
    main()
