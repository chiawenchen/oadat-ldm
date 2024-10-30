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
from torch.optim.lr_scheduler import LinearLR
from torchvision import models
from torchvision.transforms import v2
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall, BinarySpecificity

from diffusers import DDIMScheduler
from config import TrainingConfig, parse_arguments
from utils import get_last_checkpoint
import dataset


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
        # timestep = torch.tensor(timestep, dtype=torch.int64)

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
        self.transforms = v2.Compose(
            [
                v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
                v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            ]
        )
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
            transforms=self.transforms,
            inds=indices,
            noise_scheduler=self.noise_scheduler,
            label=label,
        )

    def setup(self, stage: str = None) -> None:
        print("setup datamodule...")
        if stage == "fit":
            indices_swfd = np.load(
                "/mydata/dlbirhoui/chia/oadat-ldm/train_sc_BP_indices.npy"
            )
            indices_scd = np.arange(0, 20000 * 0.95)  # 95 % for training

            # Mix SWFD and SCD datasets (80% for training, 20% for validation)
            # shuffle indices
            random.shuffle(indices_swfd)
            random.shuffle(indices_scd)

            split_idx_swfd = int(len(indices_swfd) * 0.8)
            split_idx_scd = int(len(indices_scd) * 0.8)

            # Train indices: 80% from both SWFD and SCD
            train_indices_swfd = indices_swfd[:split_idx_swfd]
            train_indices_scd = indices_scd[:split_idx_scd]

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

        elif stage == "test":
            self.test_swfd_indices = np.load(
                "/mydata/dlbirhoui/chia/oadat-ldm/test_sc_BP_indices.npy"
            )
            self.test_scd_indices = np.arange(
                20000 * 0.95, 20000
            )  # 1000 samples for testing
            self.test_swfd_obj = self.load_dataset(
                "SWFD_semicircle_RawBP.h5", "sc_BP", self.test_swfd_indices, 1
            )
            self.test_scd_obj = self.load_dataset(
                "SCD_RawBP.h5", "vc_BP", self.test_scd_indices, 0
            )
            self.test_obj = torch.utils.data.ConcatDataset(
                self.test_swfd_obj, self.test_scd_obj
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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_obj,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ResNetClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4, num_timesteps=1000, embedding_dim=128):
        super(ResNetClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim

        # Load pre-trained ResNet and adapt for grayscale input and binary classification
        self.model = models.resnet18(weights='DEFAULT')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for 1-channel input

        # Remove the original fully connected layer (fc) and replace it with our custom layer
        self.feature_dim = self.model.fc.in_features  # Typically 512
        self.model.fc = nn.Identity()  # Replace fc layer with identity to prevent its application in forward pass

        # Custom classification layer to accommodate concatenated image and timestep features
        self.fc = nn.Linear(self.feature_dim + embedding_dim, num_classes)

        # Timestep embedding layer
        self.timestep_embedding = nn.Sequential(
            nn.Embedding(num_timesteps, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.train_specificity = BinarySpecificity()
        self.val_specificity = BinarySpecificity()

    def forward(self, x, timesteps):
        # Extract features using ResNet up to the replaced fc layer
        img_features = self.model(x)  # Now outputs shape [batch_size, 512]

        # Embed the timestep and reshape for concatenation
        timestep_embed = self.timestep_embedding(timesteps)  # Shape: [batch_size, 128]

        # Concatenate image features with timestep embeddings along the last dimension
        combined_features = torch.cat([img_features, timestep_embed], dim=1)  # Shape: [batch_size, 640]

        # Pass the combined features through the final classification layer
        return self.fc(combined_features)

    def training_step(self, batch, batch_idx):
        images, labels, timesteps = batch
        outputs = self(images, timesteps)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Compute metrics
        self.train_accuracy(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_specificity(preds, labels)
        self.train_f1(preds, labels)

        # Log metrics to progress bar
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, prog_bar=False)
        self.log("train_precision", self.train_precision, prog_bar=True)
        self.log("train_recall", self.train_recall, prog_bar=True)
        self.log("train_specificity", self.train_specificity, prog_bar=False)
        self.log("train_f1", self.train_f1, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, timesteps = batch
        outputs = self(images, timesteps)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Compute metrics
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_specificity(preds, labels)
        self.val_f1(preds, labels)

        # Log metrics to progress bar
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, prog_bar=False)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_specificity", self.val_specificity, prog_bar=False)
        self.log("val_f1", self.val_f1, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels, timesteps = batch
        outputs = self(images, timesteps)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Compute metrics
        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_specificity(preds, labels)
        self.test_f1(preds, labels)

        # Log metrics to progress bar
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, prog_bar=False)
        self.log("test_precision", self.test_precision, prog_bar=True)
        self.log("test_recall", self.test_recall, prog_bar=True)
        self.log("test_specificity", self.test_specificity, prog_bar=False)
        self.log("test_f1", self.test_f1, prog_bar=True)

        return loss
    
    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = LinearLR(
            optimizer, total_iters=1, last_epoch=-1
        )
        return [optimizer], [lr_scheduler]


torch.set_float32_matmul_precision("medium")


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set up logger
    logger = WandbLogger(
        project="classifier",
        name=args.job_name,
        log_model=False,
        # config=config.__dict__,
    )

    # Set up training configuration
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    # Setup noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=1e-5,
        beta_end=5e-3,
        beta_schedule="scaled_linear",
    )

    # DataModule
    datamodule = NoisyOADATDataModule(
        data_path=args.oadat_dir,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        noise_scheduler=noise_scheduler,
    )
 
    # Model
    model = ResNetClassifier(num_classes=2)

    # Set up checkpoint callback
    ckpt_dir = os.path.join(config.output_dir, "checkpoints/classifier", args.job_name)
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
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        check_val_every_n_epoch=1,
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
