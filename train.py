## Simple script for training a diffusion model using PyTorch Lightning.

import sys, os, glob
import numpy as np
import h5py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from lightning.pytorch import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from dataclasses import dataclass
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from pathlib import Path
import dataset
import wandb


# the precision settings for matrix multiplications
torch.set_float32_matmul_precision('medium')

@dataclass
class TrainingConfig:
    image_size: int = 256  # the generated image resolution
    batch_size: int = 8  # configurable batch size
    num_epochs: int = 5  # number of epochs
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_epochs: int = 3
    save_image_epochs: int = 1
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "/mydata/dlbirhoui/chia/"  # directory to save models and images
    overwrite_output_dir: bool = True  # overwrite the old model when re-running
    seed: int = 42
    num_train_timesteps: int = 1000

class OADATDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = lambda x: torch.from_numpy(np.clip(x / np.max(x), a_min=-0.2, a_max=None))

    def prepare_data(self):
        # Download or prepare the dataset if necessary
        pass

    def setup(self, stage=None):
        print("setup datamodule...")
        indices = np.load('/mydata/dlbirhoui/chia/oadat-ldm/training_sc_BP_indices.npy')

        # Load data, create datasets
        self.dataset_obj = dataset.Dataset(
            fname_h5=os.path.join(self.data_path, 'SWFD_semicircle_RawBP.h5'),
            key='sc_BP',
            transforms=self.transforms,
            inds=indices
        )
        total_size = len(self.dataset_obj)
        train_size = int(total_size * 0.8) # train 80%
        val_size = int(total_size * 0.2) # val 15%
        test_size = total_size - train_size - val_size # test 5%

        self.train_obj, self.val_obj, self.test_obj = random_split(
            self.dataset_obj, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_obj, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_obj, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_obj, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


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
        noises = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        # generates a tensor of random integers (timesteps) ranging from 0 to num_train_timesteps - 1 for each image in the batch
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device, dtype=torch.int64)
        
        # By sampling random timesteps for each image, the model gradually learns to predict the noise at all stages of the diffusion process (from slightly noisy images to very noisy ones)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        loss = self.loss_fn(noise_preds, noises)
        
        self.log("train_loss", loss, prog_bar=True)

        # Log the current learning rate from the optimizer
        # lr = self.optimizers().param_groups[0]["lr"]
        # self.log("learning_rate", lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean_images = batch
        noises = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        # fixed seed during validation to ensure the noise is always the same
        torch.manual_seed(self.config.seed) 
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device, dtype=torch.int64)
        
        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        val_loss = self.loss_fn(noise_preds, noises)

        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.lr_warmup_epochs,
            num_training_steps=self.config.num_epochs
        )
        return [optimizer], [lr_scheduler]

    def on_validation_epoch_end(self):
        # This is where we'll sample images using DDIMPipeline at the end of each epoch
        if self.current_epoch % self.config.save_image_epochs == 0 or self.current_epoch == self.config.num_epochs - 1:
            print(f"Saving sample image at epoch {self.current_epoch}")
            pipeline = DDIMPipeline(unet=self.model, scheduler=self.noise_scheduler)
            images = pipeline(batch_size=1, generator=torch.Generator(device="cpu").manual_seed(self.config.seed)).images # self.config.batch_size
            # save sample image locally
            images[0].save(os.path.join(self.config.output_dir, f"samples/val_epoch_{self.current_epoch}.png"))
            # log sample image in wandb
            self.logger.experiment.log({"generated_images": [wandb.Image(images[0])]})


def get_last_checkpoint(checkpoint_dir):
    latest_ckpt = glob.glob(os.path.join(checkpoint_dir, "last.ckpt"))
    return latest_ckpt[-1] if latest_ckpt else None

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model on OADAT data.")
    parser.add_argument("--oadat_dir", default="oadat", type=str, help="Path to the OADAT data folder")
    parser.add_argument("--num_epochs", default=5, type=int, help="Number of epochs to train")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of dataloader workers")
    args = parser.parse_args()


    config = TrainingConfig(num_epochs=args.num_epochs, batch_size=args.batch_size)
    noise_scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps)

    datamodule = OADATDataModule(data_path=args.oadat_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    datamodule.validation_dataloader()

    # Initialize the model
    model = DiffusionModel(config=config, noise_scheduler=noise_scheduler)

    # Set up logger and checkpointing
    logger = WandbLogger(
        project="playground", 
        name="ddim-test-5", 
        log_model=False,
        config=config.__dict__
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, "checkpoints"),
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        save_last=True,
        filename='{epoch:02d}-{val_loss:.4f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1
    )

    latest_ckpt = get_last_checkpoint(os.path.join(config.output_dir, "checkpoints"))
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