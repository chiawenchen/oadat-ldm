# model.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from lightning.pytorch import LightningModule
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from torchvision.transforms import v2
from config import TrainingConfig
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import SWFD_MEAN, SWFD_STD, SCD_MEAN, SCD_STD


class DiffusionModel(LightningModule):
    def __init__(self, config: TrainingConfig, noise_scheduler: DDIMScheduler) -> None:
        super().__init__()
        self.config = config
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(
                32,
                32,
                64,
                64,
                64,
                128,
                128,
            ),  # small model one more layer
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                # add one more
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                # add one more
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.noise_scheduler = noise_scheduler
        self.loss_fn = F.mse_loss
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
        self.generator = None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps)

    def save_clean_image(
        self,
        image: torch.Tensor,
        local_path: str,
        wandb_name: str,
    ) -> None:
        plt.figure(figsize=(6, 6))
        im = plt.imshow(image.squeeze(0).cpu(), cmap="gray")
        plt.colorbar(im)
        plt.savefig(local_path, bbox_inches="tight")
        plt.close()
        self.logger.experiment.log({wandb_name: [wandb.Image(local_path)]})

    def save_images(
        self,
        images: list[torch.Tensor],
        timesteps: torch.Tensor,
        local_path: str = None,
        wandb_name: str = "wandb_name",
        transform_back: bool = False,
    ) -> None:
        if transform_back:
            # check if the image has NaN or Inf values
            for img in images:
                if torch.any(torch.isinf(img)) or torch.any(torch.isnan(img)):
                    print(f"Image contains NaN or Inf values.")
                    img[torch.isnan(img)] = 0
                    img[torch.isinf(img)] = 0

            images = [
                torch.clamp((img * SWFD_STD + SWFD_MEAN), min=-0.2, max=1)
                # torch.clamp(((img + 1.0) * 1.2 / 2.0) - 0.2, min=-0.2, max=1)
                for img in images
            ]

            # # linear scale to (0, 1)
            # images = [
            #     (img - img.min()) / (img.max() - img.min())
            #     for img in images
            # ]
        num_images = len(images)
        num_rows = 2
        num_cols = math.ceil(num_images / num_rows)
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5)
        )
        axs = axs.ravel()

        for i, img in enumerate(images):
            ax = axs[i] if num_images > 1 else axs
            if transform_back:
                im = ax.imshow(
                    img.squeeze(0).cpu(),
                    cmap="gray",
                    aspect="equal",
                    vmin=-0.2,
                    vmax=1.0,
                )
            else:
                im = ax.imshow(img.squeeze(0).cpu(), cmap="gray", aspect="equal")
            ax.axis("off")
            ax.set_title(f"t = {timesteps[i] + 1}")
            fig.colorbar(im, ax=ax)

        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.savefig(local_path, bbox_inches="tight")
        plt.close()
        self.logger.experiment.log({wandb_name: [wandb.Image(local_path)]})

    # Function to generate fixed noisy images before training
    def generate_fixed_noisy_images_for_variety_check(
        self,
        category: str,
        same_X0: bool = False,
        same_noise: bool = False,
        num_sampling: int = 5,
    ) -> None:
        local_path = os.path.join(self.config.sample_dir, f"{category}_noisy.png")
        # if os.path.exists(local_path):
        #     return

        clean_images = next(iter(self.trainer.datamodule.scd_dataloader())).to(
            self.device
        )

        if same_X0:
            clean_images = clean_images[0].unsqueeze(0).repeat(num_sampling, 1, 1, 1)
        else:
            clean_images = clean_images[:num_sampling]

        # save clean image
        self.save_images(
            clean_images,
            torch.full(
                (clean_images.size(0),), -1, dtype=torch.int64, device=self.device
            ),
            local_path=os.path.join(self.config.sample_dir, f"{category}_clean.png"),
            wandb_name=f"{category}_clean",
        )

        timesteps = torch.full(
            (clean_images.size(0),), 999, dtype=torch.int64, device=self.device
        )
        noises = torch.randn(
            clean_images.shape, generator=self.generator, device=self.device
        )
        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)

        for idx in range(len(noisy_images)):
            fixed_image_dir = f"{self.config.sample_dir}/{category}_fixed_noisy_images"
            os.makedirs(fixed_image_dir, exist_ok=True)
            noisy_image_path = os.path.join(
                fixed_image_dir,
                f"{category}_noisy_image_{idx}.pt",
            )
            torch.save(noisy_images[idx], noisy_image_path)

        # save noisy images
        self.save_images(
            noisy_images,
            timesteps,
            local_path,
            wandb_name=f"{category}_noisy",
        )

    def generate_fixed_noisy_images(self, category: str) -> None:
        local_path = os.path.join(self.config.sample_dir, f"{category}_noisy.png")
        # if os.path.exists(local_path):
        #     return
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
        self.save_clean_image(
            clean_image,
            local_path=os.path.join(self.config.sample_dir, f"{category}_clean.png"),
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
                self.config.fixed_image_paths[category],
                f"{category}_noisy_image_timestep_{t.item():03}.pt",
            )

            torch.save(noisy_image, noisy_image_path)
            noisy_images.append(noisy_image)

        # save noisy images
        self.save_images(
            noisy_images,
            timesteps,
            local_path,
            wandb_name=f"{category}_noisy",
        )

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = LinearLR(
            optimizer, total_iters=self.config.lr_warmup_epochs, last_epoch=-1
        )
        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        print("on_train_start")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)
        self.generate_fixed_noisy_images("swfd")
        self.generate_fixed_noisy_images("scd")
        self.generate_fixed_noisy_images_for_variety_check(
            "prepicked_noise", False, True
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        clean_images = batch
        noises = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        # generates a tensor of random integers (timesteps) ranging from 0 to num_train_timesteps - 1 for each image in the batch
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=self.device,
            dtype=torch.int64,
        )

        # By sampling random timesteps for each image, the model gradually learns to predict the noise at all stages of the diffusion process (from slightly noisy images to very noisy ones)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        loss = self.loss_fn(noise_preds, noises)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # fixed seed during validation to ensure the noise is always the same
        clean_images = batch
        noises = torch.randn(
            clean_images.shape, generator=self.generator, device=self.device
        )
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            dtype=torch.int64,
            generator=self.generator,
            device=self.device,
        )

        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        val_loss = self.loss_fn(noise_preds, noises)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def get_fixed_noisy_images(self, category: str) -> torch.Tensor:

        path = f"{self.config.sample_dir}/{category}_fixed_noisy_images"
        filenames = sorted(os.listdir(path))

        # Load all noisy images and stack them into a single tensor
        noisy_images = torch.stack(
            [
                torch.load(path + "/" + filename, weights_only=True).to(self.device)
                for filename in filenames
            ]
        )

        return noisy_images

    # Predicts denoised images, and saves them in WandB and locally
    def denoise_and_save_samples(
        self,
        category: str,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        epoch: int,
    ) -> None:
        self.noise_scheduler.set_timesteps(
            num_inference_steps=self.noise_scheduler.config.num_train_timesteps
        )
        denoised_images = noisy_images.clone()

        for t in range(timesteps.max(), -1, -1):  # From max timestep to 0
            active_mask = timesteps >= t  # Select images that still need processing
            if active_mask.any():
                active_noisy_images = denoised_images[active_mask]
                noise_preds = self(
                    active_noisy_images,
                    torch.full((len(active_noisy_images),), t, device=self.device),
                ).sample
                denoised = [
                    self.noise_scheduler.step(
                        noise_preds[i], t, active_noisy_images[i]
                    ).prev_sample
                    for i in range(len(active_noisy_images))
                ]
                denoised_images[active_mask] = torch.stack(denoised)

        local_path = os.path.join(
            self.config.sample_dir, f"{category}_epoch_{epoch}_denoised.png"
        )
        self.save_images(
            denoised_images,
            timesteps,
            local_path,
            wandb_name=f"{category} denoised",
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

            # 3. samples from different noise, same X_0
            noisy_images = self.get_fixed_noisy_images("prepicked_noise")
            fixed_timesteps = torch.full(
                (noisy_images.size(0),), 999, dtype=torch.int64, device=self.device
            )
            self.denoise_and_save_samples(
                "prepicked_noise", noisy_images, fixed_timesteps, self.current_epoch
            )
