# DDIM.py: Diffusion model for vanilla DM

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from lightning.pytorch import LightningModule
from diffusers import UNet2DModel, DDIMScheduler
from torchvision.transforms import v2
from config.config import TrainingConfig
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import math


class DiffusionModel(LightningModule):
    """Diffusion Model for Image Denoising using DDIM."""

    def __init__(self, config: dict, noise_scheduler: DDIMScheduler) -> None:
        """
        Initializes the diffusion model.

        Args:
            config (dict): Training configuration.
            noise_scheduler (DDIMScheduler): Scheduler for noise diffusion.
        """
        super().__init__()
        self.config = config
        self.noise_scheduler = noise_scheduler
        self.loss_fn = F.mse_loss
        self.generator = None  # Used for generating fixed noise samples

        # Initialize UNet Model
        self.model = self._build_unet()

        # Example input for visualization
        self.example_input_array = (
            torch.randn(
                self.config.batch_size,
                1,
                self.config.image_size,
                self.config.image_size,
            ),
            torch.randint(
                0, self.config.num_train_timesteps, (self.config.batch_size,)
            ),
        )

    def _build_unet(self) -> UNet2DModel:
        """Builds and returns the UNet2DModel."""
        return UNet2DModel(
            sample_size=self.config.image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 32, 64, 64, 64, 128, 128),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the UNet model."""
        return self.model(x, timesteps)

    def configure_optimizers(self):
        """Sets up the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = LinearLR(
            optimizer, total_iters=self.config.lr_warmup_epochs, last_epoch=-1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Executes a single training step."""
        clean_images = batch
        noises = torch.randn(clean_images.shape, device=self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=self.device,
            dtype=torch.int64,
        )

        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        predicted_v = self(noisy_images, timesteps).sample

        # Compute ground truth
        alpha_t = (self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1))
        sigma_t = ((1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1))
        ground_truth_v = alpha_t * noises - sigma_t * clean_images

        # Compute loss
        loss = self.loss_fn(predicted_v, ground_truth_v)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Executes a validation step to evaluate the model's performance."""
        clean_images = batch
        noises = torch.randn(
            clean_images.shape, generator=self.generator, device=self.device
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            dtype=torch.int64,
            generator=self.generator,
            device=self.device,
        )

        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        predicted_v = self(noisy_images, timesteps).sample

        # Compute ground truth
        alpha_t = (self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1))
        sigma_t = ((1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1))
        ground_truth_v = alpha_t * noises - sigma_t * clean_images

        # Compute loss
        val_loss = self.loss_fn(predicted_v, ground_truth_v)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def save_image(self, image: torch.Tensor, path: str, wandb_name: str) -> None:
        """Saves an image locally and logs it to WandB."""
        plt.figure(figsize=(6, 6))
        im = plt.imshow(image.squeeze(0).cpu(), cmap="gray")
        plt.colorbar(im)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        self.logger.experiment.log({wandb_name: [wandb.Image(path)]})

    def save_images(
        self,
        images: list[torch.Tensor],
        timesteps: torch.Tensor,
        local_path: str,
        wandb_name: str,
        transform_back: bool = False,
    ) -> None:
        """
        Saves a batch of images and logs them to WandB.

        Args:
            images (list[torch.Tensor]): List of image tensors.
            timesteps (torch.Tensor): Corresponding timesteps.
            local_path (str): Path to save the images.
            wandb_name (str): Name for WandB logging.
            transform_back (bool, optional): Whether to apply transformation. Defaults to False.
        """
        if transform_back:
            for img in images:
                if torch.any(torch.isnan(img)) or torch.any(torch.isinf(img)):
                    img[torch.isnan(img)] = 0
                    img[torch.isinf(img)] = 0
            images = [torch.clamp(((img + 1.0) * 1.2 / 2.0) - 0.2, min=-0.2, max=1) for img in images]

        num_images = len(images)
        num_rows, num_cols = 2, math.ceil(num_images / 2)
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5)
        )
        axs = axs.ravel()

        for i, img in enumerate(images):
            ax = axs[i] if num_images > 1 else axs
            im = ax.imshow(img.squeeze(0).cpu(), cmap="gray", aspect="equal",
                vmin=-0.2 if transform_back else None,
                vmax=1.0 if transform_back else None,
            )
            ax.axis("off")
            ax.set_title(f"t = {timesteps[i] + 1}")
            fig.colorbar(im, ax=ax)

        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.savefig(local_path, bbox_inches="tight")
        plt.close()
        self.logger.experiment.log({wandb_name: [wandb.Image(local_path)]})

    def generate_pure_noise_samples(
        self,
        category: str,
        same_X0: bool = False,
        same_noise: bool = False,
        num_sampling: int = 5,
    ) -> None:
        """Generates pure nois images for debugging and visualization."""
        local_path = os.path.join(self.config.paths.sample_dir, f"{category}_noisy.png")
        noises = torch.randn(
            (num_sampling, 1, self.config.image_size, self.config.image_size),
            generator=self.generator,
            device=self.device,
        )
        timesteps = torch.full(
            (noises.size(0),), 999, dtype=torch.int64, device=self.device
        )

        for idx in range(len(noises)):
            fixed_image_dir = getattr(self.config.paths.fixed_image_paths, category) 
            noisy_image_path = os.path.join(
                fixed_image_dir,
                f"{category}_noisy_image_{idx}.pt",
            )
            torch.save(noises[idx], noisy_image_path)

        # save noisy images
        self.save_images(noises, timesteps, local_path, wandb_name=f"{category}_noisy")

    def generate_noisy_samples(self, category: str, num_samples: int = 5) -> None:
        """
        Generates noisy images for debugging and visualization.

        Args:
            category (str): Category of images.
            num_samples (int): Number of images to generate.
        """
        local_path = os.path.join(self.config.paths.sample_dir, f"{category}_noisy.png")
        if category == "swfd":
            clean_images = next(iter(self.trainer.datamodule.val_dataloader())).to(
                self.device
            )
        else:
            clean_images = next(iter(self.trainer.datamodule.scd_dataloader())).to(
                self.device
            )

        clean_image = clean_images[0]
        # save clean image
        self.save_image(
            clean_image,
            path=os.path.join(
                self.config.paths.sample_dir, f"{category}_clean.png"
            ),
            wandb_name=f"{category}_clean",
        )

        # Generate linearly spaced timesteps: 0, 49, 99, ..., 999
        timesteps = torch.tensor(
            np.linspace(
                0,
                self.noise_scheduler.config.num_train_timesteps - 1,
                self.config.sample_num,
            ),
            dtype=torch.int64,
            device=self.device,
        )
        noises = torch.randn(
            clean_image.shape, generator=self.generator, device=self.device
        )
        noisy_images = []

        for idx, t in enumerate(timesteps):
            noisy_image = self.noise_scheduler.add_noise(clean_image, noises, t)
            noisy_image_path = os.path.join(
                getattr(self.config.paths.fixed_image_paths, category),
                f"{category}_noisy_image_timestep_{t.item():03}.pt",
            )

            torch.save(noisy_image, noisy_image_path)
            noisy_images.append(noisy_image)

        # save noisy images
        self.save_images(
            noisy_images, timesteps, local_path, wandb_name=f"{category}_noisy"
        )

    def on_train_start(self) -> None:
        """Executes actions at the beginning of training."""
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)
        # Generate some samples for debugging and visualization
        self.generate_noisy_samples("swfd")
        self.generate_noisy_samples("scd")
        self.generate_pure_noise_samples("prepicked_noise", False, True)

    def get_fixed_noisy_images(self, category: str) -> torch.Tensor:
        """Loads and returns fixed noisy images from storage."""
        path = getattr(self.config.paths.fixed_image_paths, category) 
        filenames = sorted(os.listdir(path))
        return torch.stack(
            [
                torch.load(os.path.join(path, filename), weights_only=True).to(self.device)
                for filename in filenames
            ]
        )

    def denoise_and_save_samples(
        self,
        category: str,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        epoch: int,
    ) -> None:
        """Performs iterative denoising and saves results."""
        self.noise_scheduler.set_timesteps(
            num_inference_steps=self.noise_scheduler.config.num_train_timesteps
        )
        denoised_images = noisy_images.clone()

        for t in range(timesteps.max(), -1, -1):  # From max timestep to 0
            active_mask = timesteps >= t
            if active_mask.any():
                active_noisy_images = denoised_images[active_mask]
                predicted_v = self(
                    active_noisy_images,
                    torch.full((len(active_noisy_images),), t, device=self.device),
                ).sample
                denoised_images[active_mask] = torch.stack(
                    [
                        self.noise_scheduler.step(
                            predicted_v[i], t, active_noisy_images[i]
                        ).prev_sample
                        for i in range(len(active_noisy_images))
                    ]
                )

        local_path = os.path.join(
            self.config.paths.sample_dir, f"{category}_epoch_{epoch}_denoised.png"
        )
        self.save_images(
            denoised_images,
            timesteps,
            local_path,
            wandb_name=f"{category}_denoised",
            transform_back=True,
        )

    def on_validation_epoch_end(self) -> None:
        if (
            self.current_epoch == 0
            or self.current_epoch % self.config.save_image_epochs == 0
            or self.current_epoch == self.config.num_epochs - 1
        ):
            # Define the linearly spaced timesteps
            fixed_timesteps = torch.tensor(
                np.linspace(
                    0,
                    self.noise_scheduler.config.num_train_timesteps - 1,
                    self.config.sample_num,
                ),
                dtype=torch.int64,
                device=self.device,
            )

            # 1. Use the fixed noisy images for the validation set
            val_noisy_images = self.get_fixed_noisy_images("swfd")
            self.denoise_and_save_samples(
                "swfd", val_noisy_images, fixed_timesteps, self.current_epoch
            )

            # 2. Use the fixed noisy images for the SCD set
            scd_noisy_images = self.get_fixed_noisy_images("scd")
            self.denoise_and_save_samples(
                "scd", scd_noisy_images, fixed_timesteps, self.current_epoch
            )

            # 3. samples from pure gaussian noises
            noisy_images = self.get_fixed_noisy_images("prepicked_noise")
            fixed_timesteps = torch.full(
                (noisy_images.size(0),), 999, dtype=torch.int64, device=self.device
            )
            self.denoise_and_save_samples(
                "prepicked_noise", noisy_images, fixed_timesteps, self.current_epoch
            )
