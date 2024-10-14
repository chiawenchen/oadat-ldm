# datamodule.py

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning.pytorch import LightningDataModule
import dataset

class OADATDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, num_workers: int = 4) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = v2.Compose([
            v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
            v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        ])

    def prepare_data(self) -> None:
        # Download or prepare the dataset if necessary
        pass

    def load_dataset(self, fname_h5: str, key: str, indices: list[int]) -> dataset.Dataset:
        return dataset.Dataset(
            fname_h5=os.path.join(self.data_path, fname_h5),
            key=key,
            transforms=self.transforms,
            inds=indices
        )

    def setup(self, stage: str = None) -> None:
        print("setup datamodule...")
        if stage == "fit":
            indices = np.load('/mydata/dlbirhoui/chia/oadat-ldm/train_sc_BP_indices.npy')
            indices = indices
            split_idx = int(len(indices) * 0.8)
            self.train_indices, self.val_indices = indices[:split_idx], indices[split_idx:]
            self.train_obj = self.load_dataset('SWFD_semicircle_RawBP.h5', 'sc_BP', self.train_indices)
            self.val_obj = self.load_dataset('SWFD_semicircle_RawBP.h5', 'sc_BP', self.val_indices)


        elif stage == "test":
            self.test_indices = np.load('/mydata/dlbirhoui/chia/oadat-ldm/test_sc_BP_indices.npy')
            self.test_obj = self.load_dataset('SWFD_semicircle_RawBP.h5', 'sc_BP', self.test_indices)

        self.scd_obj = self.load_dataset('SCD_RawBP.h5', 'vc_BP', [0])
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_obj, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_obj, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_obj, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def scd_dataloader(self) -> DataLoader:
        return DataLoader(self.scd_obj, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

