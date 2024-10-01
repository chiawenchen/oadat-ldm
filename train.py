## Simple script for training a diffusion model using PyTorch Lightning.

import sys, os, glob
import numpy as np
import h5py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from dataclasses import dataclass
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from pathlib import Path
import dataset


@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 8  # configurable batch size
    eval_batch_size = 8
    num_epochs = 5
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = "ddpm-samples"
    overwrite_output_dir = True
    seed = 0


class DiffusionModel(LightningModule):
    def __init__(self, config, noise_scheduler):
        super().__init__()
        self.config = config
        self.model = UNet2DModel(
            sample_size=config.image_size,  # target image resolution
            in_channels=1,  # grayscale input (1 channel)
            out_channels=1,  # grayscale output
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
        )
        self.noise_scheduler = noise_scheduler
        self.loss_fn = F.mse_loss

    def forward(self, x, timesteps):
        return self.model(x, timesteps)

    def training_step(self, batch, batch_idx):
        clean_images = batch
        noise = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device, dtype=torch.int64)
        
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = self(noisy_images, timesteps)
        loss = self.loss_fn(noise_pred, noise)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean_images = batch
        noise = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device, dtype=torch.int64)
        
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = self(noisy_images, timesteps)
        val_loss = self.loss_fn(noise_pred, noise)

        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=len(self.trainer.datamodule.train_dataloader()) * self.config.num_epochs
        )
        return [optimizer], [lr_scheduler]

    def on_epoch_end(self):
        # This is where we'll sample images using DDPMPipeline at the end of each epoch
        if self.current_epoch % self.config.save_image_epochs == 0 or self.current_epoch == self.config.num_epochs - 1:
            pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)
            images = pipeline(batch_size=self.config.eval_batch_size, generator=torch.Generator(device="cpu").manual_seed(self.config.seed)).images

            # Create image grid and save the images
            grid = make_image_grid(images, rows=4, cols=4)
            save_dir = os.path.join(self.config.output_dir, "samples")
            os.makedirs(save_dir, exist_ok=True)
            grid.save(f"{save_dir}/{self.current_epoch:04d}.png")


def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model on OADAT data.")
    parser.add_argument("--oadat_dir", default="oadat", type=str, help="Path to the OADAT data folder")
    parser.add_argument("--num_epochs", default=5, type=int, help="Number of epochs to train")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of dataloader workers")
    args = parser.parse_args()

    # Load the dataset and data loader
    dataset_obj = dataset.Dataset(
        fname_h5=os.path.join(args.oadat_dir, 'SWFD_semicircle_RawBP.h5'),
        key='sc_BP',
        transforms=lambda x: torch.from_numpy(np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
    )
    train_dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Create the training configuration
    config = TrainingConfig(num_epochs=args.num_epochs, train_batch_size=args.batch_size)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initialize the model
    model = DiffusionModel(config=config, noise_scheduler=noise_scheduler)

    # Set up logger and checkpointing
    logger = TensorBoardLogger("logs", name="diffusion_model")
    logger = WandbLogger(project="playground")
    checkpoint_callback = ModelCheckpoint(
        dirpath="/mydata/dlbirhoui/chia/checkpoints",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=True
    )

    # Set up Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()