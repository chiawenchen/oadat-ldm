## Simple script for training a diffusion model using PyTorch Lightning.

import os, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim.lr_scheduler import LinearLR

from lightning.pytorch import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from lightning.pytorch.loggers import WandbLogger

from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
import dataset
import wandb

import argparse
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # default hyperparameters
    image_size: int = 256 
    batch_size: int = 32
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_epochs: int = 5
    save_image_epochs: int = 1
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "/mydata/dlbirhoui/chia/"  # directory to save models and images
    seed: int = 42
    num_train_timesteps: int = 1000
    sample_dir: str = None


# the precision settings for matrix multiplications
torch.set_float32_matmul_precision('medium')

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
            indices = indices[:100]
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

class DiffusionModel(LightningModule):
    def __init__(self, config: TrainingConfig, noise_scheduler: DDIMScheduler) -> None:
        super().__init__()
        self.config = config
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=1, 
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 64, 128, 128, 256, 256),
            # block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
        )
        self.noise_scheduler = noise_scheduler
        self.loss_fn = F.mse_loss

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = LinearLR(
            optimizer,
            total_iters=self.config.lr_warmup_epochs, 
            last_epoch=-1
        )
        return [optimizer], [lr_scheduler]

 

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        clean_images = batch
        noises = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        # generates a tensor of random integers (timesteps) ranging from 0 to num_train_timesteps - 1 for each image in the batch
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device, dtype=torch.int64)
        
        # By sampling random timesteps for each image, the model gradually learns to predict the noise at all stages of the diffusion process (from slightly noisy images to very noisy ones)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        loss = self.loss_fn(noise_preds, noises)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
 

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        clean_images = batch
        noises = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        # fixed seed during validation to ensure the noise is always the same
        # torch.manual_seed(self.config.seed) 
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device, dtype=torch.int64)
        
        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        val_loss = self.loss_fn(noise_preds, noises)
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return val_loss
 
    def save_images(self, clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor, epoch: int, category: str) -> None:

        # convert tensors to PIL image
        noisy = v2.ToPILImage()(noisy.cpu())
        denoised = v2.ToPILImage()(denoised.cpu())

        # save images locally
        noisy.save(os.path.join(self.config.sample_dir, f"epoch_{epoch}_{category}_noisy.png"))
        denoised.save(os.path.join(self.config.sample_dir, f"epoch_{epoch}_{category}_denoised.png"))

        # save images in wandb
        self.logger.experiment.log({f"{category}_noisy_image": [wandb.Image(noisy)]})
        self.logger.experiment.log({f"{category}_denoised_image": [wandb.Image(denoised)]})

        # save clean image only for the first 3 epochs
        if epoch < 3:
            clean = v2.ToPILImage()(clean.cpu())
            clean.save(os.path.join(self.config.sample_dir, f"epoch_{epoch}_{category}_clean.png"))
            self.logger.experiment.log({f"{category}_clean_image": [wandb.Image(clean)]})

    # Generates noisy images, predicts denoised images, and saves them in WandB and locally
    def generate_and_save_samples(self, category: str, images: torch.Tensor, timesteps: torch.Tensor) -> None:
        noises = torch.randn(images.shape, device=self.device)
        noisy_images = self.noise_scheduler.add_noise(images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        self.noise_scheduler.set_timesteps(num_inference_steps=self.noise_scheduler.config.num_train_timesteps)
        denoised_image = self.noise_scheduler.step(noise_preds[0], timesteps[0], noisy_images[0]).pred_original_sample
        self.save_images(images[0], noisy_images[0], denoised_image, self.current_epoch, category)

    def on_validation_epoch_end(self) -> None:
        # sample images from validation set, SCD dataset, and random noise
        if self.current_epoch % self.config.save_image_epochs == 0 or self.current_epoch == self.config.num_epochs - 1:

            # fixed seed during validation to ensure the noise is always the same
            torch.manual_seed(self.config.seed)
            # prevent images from being too noisy
            MAX_NOISE_LEVEL = 0.5
            timesteps = torch.randint(0, int(self.noise_scheduler.config.num_train_timesteps * MAX_NOISE_LEVEL), (self.config.batch_size,), device=self.device, dtype=torch.int64)

            # 1. sample an image from validation set (add less than 50% noise)
            clean_images = next(iter(self.trainer.datamodule.val_dataloader())).to(self.device)
            self.generate_and_save_samples("val", clean_images, timesteps)

            # 2. sample an image from SCD dataset (add less than 50% noise)
            scd_images = next(iter(self.trainer.datamodule.scd_dataloader())).to(self.device)
            self.generate_and_save_samples("scd", scd_images, timesteps)

            
            # 3. sample an image from random noise
            pipeline = DDIMPipeline(unet=self.model, scheduler=self.noise_scheduler)
            images = pipeline(batch_size=1, generator=torch.Generator(device="cpu").manual_seed(self.config.seed)).images
            # save sample image locally and in wandb
            images[0].save(os.path.join(self.config.sample_dir, f"epoch_{self.current_epoch}_random.png"))
            self.logger.experiment.log({"random_noise": [wandb.Image(images[0])]})
 

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diffusion model on OADAT data.")
    parser.add_argument("--oadat_dir", default="oadat", type=str, help="Path to the OADAT data folder")
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of epochs to train")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of dataloader workers")
    parser.add_argument("--job_name", default='ddim-test', type=str, help="Job name")
    return parser.parse_args()

def get_last_checkpoint(checkpoint_dir: str) -> str:
    latest_ckpt = glob.glob(os.path.join(checkpoint_dir, "last.ckpt"))
    return latest_ckpt[-1] if latest_ckpt else None

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
        num_workers=args.num_workers
    )

    # Set up noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps
    )

    # Initialize the model
    model = DiffusionModel(
        config=config, 
        noise_scheduler=noise_scheduler
    )

    # Set up logger
    logger = WandbLogger(
        project="playground", 
        name=args.job_name, 
        log_model=False,
        config=config.__dict__
    )

    # Set up sample image saving path
    sample_dir = os.path.join(config.output_dir, "samples", args.job_name)
    os.makedirs(sample_dir, exist_ok=True)
    config.sample_dir = sample_dir

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
        filename='{epoch:02d}-{val_loss:.4f}'
    )

    # Set up Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
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