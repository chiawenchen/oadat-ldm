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
from config.config import ClassifierConfig, parse_arguments
from utils import get_last_checkpoint, get_named_beta_schedule, transforms
import dataset

from models.UnetClassifier import UnetAttentionClassifier


class NoisyDataset(dataset.Dataset):
    def __init__(
        self,
        fname_h5,
        key,
        transforms,
        inds,
        noise_scheduler,
        label,
        shuffle=False,
        **kwargs,
    ):
        super().__init__(fname_h5, key, transforms, inds, shuffle=shuffle, **kwargs)
        self.noise_scheduler = noise_scheduler
        self.label = label
        self.num_timesteps = noise_scheduler.config.num_train_timesteps

    def __getitem__(self, index):
        # Load the base image data as before
        with h5py.File(self.fname_h5, "r") as fh:
            x = fh[self.key][index, ...]
            x = x[None, ...]  # Add a channel dimension [1, H, W]
            if self.transforms is not None:
                x = self.transforms(x)

        # Convert image to a torch tensor if it's not already
        x = torch.tensor(x, dtype=torch.float32)

        # Sample a random timestep
        timestep = torch.randint(0, self.num_timesteps, (1,), dtype=torch.int64)

        # Add noise to the image based on the timestep
        noise = torch.randn_like(x).unsqueeze(0)
        noisy_x = self.noise_scheduler.add_noise(x, noise, timestep).squeeze(0)

        # Prepare the label and timestep for return
        label = torch.tensor(self.label, dtype=torch.long)

        return noisy_x, label, timestep.item()


class NoisyOADATDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int = 4,
        noise_scheduler: DDIMScheduler = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_scheduler = noise_scheduler

    def load_dataset(
        self,
        fname_h5: str,
        key: str,
        indices: list[int],
        label: int,
    ) -> dataset.Dataset:
        return NoisyDataset(
            fname_h5=os.path.join(self.data_path, fname_h5),
            key=key,
            transforms=transforms,
            inds=indices,
            noise_scheduler=self.noise_scheduler,
            label=label,
        )

    def setup(self, stage: str = None) -> None:
        print("setup datamodule...")
        if stage == "fit":
            indices_swfd = np.load(
                "/mydata/dlbirhoui/chia/oadat-ldm/config/train_sc_BP_indices.npy"
            )
            indices_scd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_train_indices.npy")
            # Mix SWFD and SCD datasets (80% for training, 20% for validation)
            split_idx_swfd = int(len(indices_swfd) * 0.8)
            split_idx_scd = int(len(indices_scd) * 0.8)

            # Train indices: 80% from both SWFD and SCD
            train_indices_swfd = indices_swfd[:split_idx_swfd]
            train_indices_scd = indices_scd[:split_idx_scd]

            print("num of swfd: ", len(train_indices_swfd))
            print("num of scd: ", len(train_indices_scd))

            # Val indices: 20% from both SWFD and SCD
            val_indices_swfd = indices_swfd[split_idx_swfd:]
            val_indices_scd = indices_scd[split_idx_scd:]

            # Load datasets
            self.train_obj_swfd = self.load_dataset(
                "SWFD_semicircle_RawBP.h5", "sc_BP", train_indices_swfd, 1
            )
            self.train_obj_scd = self.load_dataset(
                "SCD_RawBP.h5", "vc_BP", train_indices_scd, 0
            )
            self.val_obj_swfd = self.load_dataset(
                "SWFD_semicircle_RawBP.h5", "sc_BP", val_indices_swfd, 1
            )
            self.val_obj_scd = self.load_dataset(
                "SCD_RawBP.h5", "vc_BP", val_indices_scd, 0
            )

            # Combine SWFD and SCD datasets for both training and validation
            self.train_obj = torch.utils.data.ConcatDataset(
                [self.train_obj_swfd, self.train_obj_scd]
            )
            self.val_obj = torch.utils.data.ConcatDataset(
                [self.val_obj_swfd, self.val_obj_scd]
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_obj,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_obj,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

torch.set_float32_matmul_precision("medium")


def main():

    # Parse command-line arguments
    args = parse_arguments()

    # Set up training configuration
    config = ClassifierConfig()

    # Set up logger
    logger = WandbLogger(
        project="classifier",
        name=args.job_name,
        log_model=False,
        config=config.__dict__
    )

    # Setup noise scheduler
    noise_scheduler = get_named_beta_schedule(args.noise_schedule, config.num_train_timesteps)

    # DataModule
    datamodule = NoisyOADATDataModule(
        data_path=args.oadat_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        noise_scheduler=noise_scheduler,
    )
 
    # Model
    if args.classifier == 'resnet':
        model = ResNetClassifier(num_classes=2)
    elif args.classifier == 'unet':
        model = UnetClassifier(
            num_timesteps=config.num_train_timesteps,
        )
    elif args.classifier == 'unet_attention':
        model = UnetAttentionClassifier(
            **config.__dict__
        )

    # Set up checkpoint callback
    ckpt_dir = os.path.join("/mydata/dlbirhoui/chia/checkpoints/classifier", args.job_name)
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

    # Trainer with Wandb logger
    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        precision=16,
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
