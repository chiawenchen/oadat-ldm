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
from dataset import LabeledDataset

# from models.VAE import VAE
from models.CVAE import CVAE

torch.set_float32_matmul_precision("medium")


class LabeledOADATDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
        mix_swfd_scd: bool,
        indices_swfd: list[int],
        indices_scd: list[int]
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mix_swfd_scd = mix_swfd_scd
        self.indices_swfd = indices_swfd
        self.indices_scd = indices_scd

        print('train swfd size: ', len(self.indices_swfd))
        print('train scd size: ', len(self.indices_scd))
        print('mix_swfd_scd: ', self.mix_swfd_scd)

    def load_dataset(
        self,
        fname_h5: str,
        key: str,
        indices: list[int],
        label: int,
    ) -> LabeledDataset:
        return LabeledDataset(
            fname_h5=os.path.join(self.data_path, fname_h5),
            key=key,
            transforms=transforms,
            inds=indices,
            label=label,
        )

    def setup(self, stage: str = None) -> None:
        print("setup datamodule...")
        if stage == "fit":
            if self.mix_swfd_scd:
                indices_swfd = self.indices_swfd
                indices_scd = self.indices_scd
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

def main():

    # Parse command-line arguments
    args = parse_arguments()

    # sample directory
    sample_dir = os.path.join("/mydata/dlbirhoui/chia/samples/cvae", args.job_name)
    os.makedirs(sample_dir, exist_ok=True)

    # vae model
    model = CVAE(sample_dir=sample_dir)

    # Set up logger
    logger = WandbLogger(
        project="cvae",
        name=args.job_name,
        log_model=False,
        # config=config.__dict__
    )

    # DataModule
    indices_scd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_train_indices.npy")
    indices_swfd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/train_sc_BP_indices.npy")
    datamodule = LabeledOADATDataModule(
        data_path=args.oadat_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_swfd_scd=args.mix_swfd_scd,
        indices_swfd=indices_swfd,
        indices_scd=indices_scd
    )

    # Set up checkpoint callback
    ckpt_dir = os.path.join("/mydata/dlbirhoui/chia/checkpoints/cvae", args.job_name)
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
        check_val_every_n_epoch=1,
        num_sanity_val_steps=1,
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
