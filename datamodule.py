# datamodule.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning.pytorch import LightningDataModule
import dataset
from diffusers import DDIMScheduler
from utils import transforms


class OADATDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int = 4,
        mix_swfd_scd: bool = False,
        in_channel: int = 1,
        indices_swfd: list[int] = np.load(
                "/mydata/dlbirhoui/chia/oadat-ldm/config/train_sc_BP_indices.npy"
            ),
        indices_scd: list[int] = np.arange(0, 20000)
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


    def prepare_data(self) -> None:
        # Download or prepare the dataset if necessary
        pass

    def load_dataset(
        self, fname_h5: str, key: str, indices: list[int]
    ) -> dataset.Dataset:
        return dataset.Dataset(
            fname_h5=os.path.join(self.data_path, fname_h5),
            key=key,
            transforms=transforms,
            inds=indices,
        )

    def setup(self, stage: str = None) -> None:
        print("setup datamodule...")
        if stage == "fit":
            if self.mix_swfd_scd:
                # Mix SWFD and SCD datasets (80% for training, 20% for validation)
                split_idx_swfd = int(len(self.indices_swfd) * 0.8)
                split_idx_scd = int(len(self.indices_scd) * 0.8)

                # Train indices: 80% from both SWFD and SCD
                train_indices_swfd = self.indices_swfd[:split_idx_swfd]
                train_indices_scd = self.indices_scd[:split_idx_scd]

                # Val indices: 20% from both SWFD and SCD
                val_indices_swfd = self.indices_swfd[split_idx_swfd:]
                val_indices_scd = self.indices_scd[split_idx_scd:]

                # Load datasets
                self.train_obj_swfd = self.load_dataset(
                    "SWFD_semicircle_RawBP.h5",
                    "sc_BP",
                    train_indices_swfd
                )
                self.train_obj_scd = self.load_dataset(
                    "SCD_RawBP.h5", "vc_BP", train_indices_scd
                )
                self.val_obj_swfd = self.load_dataset(
                    "SWFD_semicircle_RawBP.h5",
                    "sc_BP",
                    val_indices_swfd
                )
                self.val_obj_scd = self.load_dataset(
                    "SCD_RawBP.h5", "vc_BP", val_indices_scd
                )

                # Combine SWFD and SCD datasets for both training and validation
                self.train_obj = torch.utils.data.ConcatDataset(
                    [self.train_obj_swfd, self.train_obj_scd]
                )
                self.val_obj = torch.utils.data.ConcatDataset(
                    [self.val_obj_swfd, self.val_obj_scd]
                )

            else:
                print("only trained on swfd")
                split_idx = int(len(self.indices_swfd) * 0.8)
                self.train_indices, self.val_indices = (
                    self.indices_swfd[:split_idx],
                    self.indices_swfd[split_idx:],
                )
                self.train_obj = self.load_dataset(
                    "SWFD_semicircle_RawBP.h5",
                    "sc_BP",
                    self.train_indices
                )
                self.val_obj = self.load_dataset(
                    "SWFD_semicircle_RawBP.h5",
                    "sc_BP",
                    self.val_indices
                )

        self.scd_obj = self.load_dataset(
            "SCD_RawBP.h5",
            "vc_BP",
            self.indices_scd[:10]
        )

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_obj,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_obj,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_obj,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def scd_dataloader(self) -> DataLoader:
        return DataLoader(
            self.scd_obj,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
