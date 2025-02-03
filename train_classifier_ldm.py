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
from config.parser import parse_arguments
from utils import (
    get_last_checkpoint,
    get_named_beta_schedule,
    load_config_from_yaml,
    transforms,
)
import dataset

from models.UnetClassifier import UnetAttentionClassifier

# Set precision for matrix multiplications
torch.set_float32_matmul_precision("medium")


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
        print("len inds: ", len(self.inds))

    def __getitem__(self, index):
        # Load the base image data as before
        with h5py.File(self.fname_h5, "r") as fh:
            x = fh[self.key][self.inds[index], ...]
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
        indices_scd: str,
        indices_swfd: str,
        noise_scheduler: DDIMScheduler = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_scheduler = noise_scheduler
        self.indices_swfd = indices_swfd
        self.indices_scd = indices_scd

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
            # Mix SWFD and SCD datasets (80% for training, 20% for validation)
            split_idx_swfd = int(len(self.indices_swfd) * 0.8)
            split_idx_scd = int(len(self.indices_scd) * 0.8)

            # Train indices: 80% from both SWFD and SCD
            train_indices_swfd = self.indices_swfd[:split_idx_swfd]
            train_indices_scd = self.indices_scd[:split_idx_scd]

            print("num of swfd: ", len(train_indices_swfd))
            print("num of scd: ", len(train_indices_scd))

            # Val indices: 20% from both SWFD and SCD
            val_indices_swfd = self.indices_swfd[split_idx_swfd:]
            val_indices_scd = self.indices_scd[split_idx_scd:]

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


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load training configuration from YAML
    config = load_config_from_yaml(args.config_path)

    # Ensure dataset paths exist before loading
    if not os.path.exists(config.dataset.scd_train_indices):
        raise FileNotFoundError(
            f"Dataset file not found: {config.dataset.scd_train_indices}"
        )
    if not os.path.exists(config.dataset.swfd_train_indices):
        raise FileNotFoundError(
            f"Dataset file not found: {config.dataset.swfd_train_indices}"
        )

    scd_train_indices = np.load(config.dataset.scd_train_indices)
    swfd_train_indices = np.load(config.dataset.swfd_train_indices)

    # Initialize DataModule
    datamodule = NoisyOADATDataModule(
        data_path=args.oadat_dir,
        batch_size=config.batch_size,
        indices_scd=scd_train_indices,
        indices_swfd=swfd_train_indices,
        noise_scheduler=get_named_beta_schedule(
            config.noise_schedule, config.model.num_train_timesteps
        ),
        num_workers=config.num_workers,
    )

    # Initialize the classifier model based on config
    model = UnetAttentionClassifier(**vars(config.model))

    # Set up WandB logger
    logger = WandbLogger(
        project=config.wandb.project_name,
        name=config.wandb.job_name,
        log_model=False,
        config=vars(config),
    )

    # Set up checkpoint callback
    os.makedirs(config.ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.ckpt_dir,
        save_top_k=config.checkpointing.save_top_k,
        monitor=config.checkpointing.monitor_metric,
        mode=config.checkpointing.monitor_mode,
        save_weights_only=False,
        save_last=config.checkpointing.save_last,
        filename=config.checkpointing.filename_format,
    )

    # Set up Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        precision=16,
        num_sanity_val_steps=1,
    )

    # Load the latest checkpoint if available
    latest_ckpt = get_last_checkpoint(config.ckpt_dir)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        trainer.fit(model, datamodule, ckpt_path=latest_ckpt)
    else:
        print("No checkpoint found, starting from scratch.")
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
