# LDM.py: Latent Diffusion Model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from lightning.pytorch import LightningModule
from diffusers import UNet2DModel, DDIMScheduler
from torchvision.transforms import v2
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from models.VAE import VAE

class LDM(LightningModule):
    def __init__(self, config: dict, noise_scheduler: DDIMScheduler, vae: VAE) -> None:
        super().__init__()
        self.sample_size = config.image_size // 2 ** (config.num_down_blocks - 1)
        self.in_channels = config.latent_channels
        self.out_channels = self.in_channels
        self.config = config
        self.noise_scheduler = noise_scheduler
        self.loss_fn = F.mse_loss
        self.vae = vae

        # Example input for visualization
        self.example_input_array = (
            torch.randn(
                self.config.batch_size,
                self.in_channels,
                self.sample_size,
                self.sample_size,
            ),
            torch.randint(
                0, self.config.num_train_timesteps, (self.config.batch_size,)
            ),
        )
        self.generator = None

        # Initialize UNet Model for the diffusion process
        self.model = self._build_unet()

        # Freeze the VAE parameters to keep the encoder/decoder static
        for param in self.vae.parameters():
            param.requires_grad = False

    def _build_unet(self) -> UNet2DModel:
        """Builds and returns the UNet model used for diffusion."""
        return UNet2DModel(
            sample_size=self.sample_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            layers_per_block=self.config.layers_per_block,
            center_input_sample=False,
            block_out_channels=self.config.block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the UNet model."""
        return self.model(x, timesteps)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        # Optimize only the diffusion model parameters
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        lr_scheduler = LinearLR(
            optimizer, total_iters=self.config.lr_warmup_epochs, last_epoch=-1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        posterior = self.vae.vae.encode(batch).latent_dist
        latents = torch.sigmoid(posterior.sample()) * 2.0 - 1.0
        noises = torch.randn(latents.shape, device=self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch.shape[0],),
            device=self.device,
            dtype=torch.int64,
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noises, timesteps)
        predicted_v = self(noisy_latents, timesteps).sample

        # Compute ground truth v
        alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        ground_truth_v = alpha_t * noises - sigma_t * latents

        # Compute loss
        loss = self.loss_fn(predicted_v, ground_truth_v)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        posterior = self.vae.vae.encode(batch).latent_dist
        latents = torch.sigmoid(posterior.sample()) * 2.0 - 1.0

        # Fix generator during validation to ensure the noise is always the same
        noises = torch.randn(
            latents.shape, generator=self.generator, device=self.device
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch.shape[0],),
            dtype=torch.int64,
            generator=self.generator,
            device=self.device,
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noises, timesteps)
        predicted_v = self(noisy_latents, timesteps).sample

        # Compute ground truth
        alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        ground_truth_v = alpha_t * noises - sigma_t * latents

        # Compute loss
        val_loss = self.loss_fn(predicted_v, ground_truth_v)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

### Below is for debugging and visualization at the end of every n epochs ###
    def save_image(
        self,
        image: torch.Tensor,
        local_path: str,
        wandb_name: str,
    ) -> None:
        """Saves a single grayscale image locally and logs it to WandB."""
        plt.figure(figsize=(6, 6))
        im = plt.imshow(image.squeeze(0).cpu().detach().numpy(), cmap="gray", vmin=-1.0, vmax=1.0)
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
        """Saves multiple images as a grid and logs them to WandB."""
        if transform_back:
            # Ensure no NaN or Inf values exist in the images
            for img in images:
                if torch.any(torch.isinf(img)) or torch.any(torch.isnan(img)):
                    print(f"Image contains NaN or Inf values.")
                    img[torch.isnan(img)] = 0
                    img[torch.isinf(img)] = 0
            images = [torch.clamp(((torch.clamp(img, min=-1.0, max=1.0) + 1.0) * 1.2 / 2.0) - 0.2, min=-0.2, max=1) for img in images]

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
                im = ax.imshow(img.squeeze(0).cpu().detach().numpy(), cmap="gray", aspect="equal", vmin=-0.2, vmax=1.0)
            else:
                im = ax.imshow(img.squeeze(0).cpu().detach().numpy(), cmap="gray", aspect="equal")
            ax.axis("off")
            ax.set_title(f"t = {timesteps[i] + 1}")
            fig.colorbar(im, ax=ax)

        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.savefig(local_path, bbox_inches="tight")
        plt.close()
        self.logger.experiment.log({wandb_name: [wandb.Image(local_path, caption=f"Epoch {self.current_epoch}")]})

    def generate_noisy_samples(self, category: str) -> None:
        """Generates noisy samples from clean images using the diffusion process."""
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
        # Save original clean image
        self.save_image(
            clean_image,
            local_path=os.path.join(self.config.paths.sample_dir, f"{category}_clean.png"),
            wandb_name=f"{category}_clean",
        )

        # Encode and reconstruct the clean image through VAE
        reconstructed, _ = self.vae.forward(clean_image.unsqueeze(0))

        # Save reconstructed clean image
        self.save_image(
            reconstructed.squeeze(0),
            local_path=os.path.join(self.config.paths.sample_dir, f"{category}_clean_reconstructed.png"),
            wandb_name=f"{category}_clean_reconstructed",
        )

        # Generate timesteps between 0 and max diffusion step
        timesteps = torch.tensor(
            np.linspace(
                0,
                self.noise_scheduler.config.num_train_timesteps - 1,
                self.config.sample_num,
            ),
            dtype=torch.int64,
            device=self.device,
        )
        posterior = self.vae.vae.encode(clean_image.unsqueeze(0)).latent_dist

        z = torch.sigmoid(posterior.sample()) * 2.0 - 1.0
        noises = torch.randn(z.shape, generator=self.generator, device=self.device)

        noisy_latents = []
        for idx, t in enumerate(timesteps):
            noisy_latent = self.noise_scheduler.add_noise(z, noises, t).squeeze(0)
            noisy_latents.append(noisy_latent)

            noisy_latent_path = getattr(self.config.paths.fixed_image_paths, category) 
            noisy_latent_path = os.path.join(noisy_latent_path, f"{category}_noisy_image_timestep_{t.item():03}.pt",)
            torch.save(noisy_latent, noisy_latent_path)

    def on_train_start(self) -> None:
        print("on_train_start")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)
        # Generate some samples for debugging and visualization
        self.generate_noisy_samples("swfd")
        self.generate_noisy_samples("scd")


    def get_fixed_noisy_images(self, category: str) -> torch.Tensor:
        path = getattr(self.config.paths.fixed_image_paths, category) 
        filenames = sorted(os.listdir(path))

        # Load all noisy images and stack them into a single tensor
        noisy_images = torch.stack(
            [ torch.load(path + "/" + filename, weights_only=True).to(self.device) for filename in filenames ]
        )
        return noisy_images

    def denoise_and_save_samples(
        self,
        category: str,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        epoch: int,
    ) -> None:
        """Performs iterative denoising and saves results."""
        self.noise_scheduler.set_timesteps(
            num_inference_steps=self.noise_scheduler.config.num_train_timesteps
        )

        denoised_latents = noisy_latents.clone()
        for t in range(timesteps.max(), -1, -1):  # From max timestep to 0
            active_mask = timesteps >= t  # Select images that still need processing
            if active_mask.any():
                active_latents = denoised_latents[active_mask]
                # Predict v using the model
                predicted_v = self(
                    active_latents,
                    torch.full((len(active_latents),), t, device=self.device),
                ).sample

                # Use the scheduler to step back
                step_result = [
                    self.noise_scheduler.step(
                        predicted_v[i], t, active_latents[i]
                    ).prev_sample
                    for i in range(len(active_latents))
                ]

                # Update denoised images
                denoised_latents[active_mask] = torch.stack(step_result)

        for idx, latent in enumerate(denoised_latents):
            print(f"----------After Denoising - Image: {idx}-----------")
            latent_flatten = latent.flatten()
            print('min: ', latent_flatten.min().item())
            print('q25: ', torch.quantile(latent_flatten, 0.25).item())
            print('q50: ', torch.quantile(latent_flatten, 0.50).item())
            print('q75: ', torch.quantile(latent_flatten, 0.75).item())
            print('max: ', latent_flatten.max().item())
            print('std: ', latent_flatten.std().item())
            print('mean: ', latent_flatten.mean().item())
            print(" ")

        # Decode from latent space to image space
        denoised_latents = (denoised_latents + 1.0) / 2.0
        denoised_images = self.vae.vae.decode(denoised_latents).sample

        # Save the denoised images
        self.save_images(
            denoised_images,
            timesteps,
            local_path=os.path.join(self.config.paths.sample_dir, f"{category}_epoch_{epoch}_denoised.png"),
            wandb_name=f"{category} denoised",
            transform_back=True,
        )

    def on_validation_epoch_end(self) -> None:
        if (
            self.current_epoch == 0
            or self.current_epoch % self.config.save_image_epochs == 0
            or self.current_epoch == self.config.num_epochs - 1
        ):
            # Define the timesteps
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
            print('swfd success.')

            # 2. Use the fixed noisy images for the SCD set
            scd_noisy_images = self.get_fixed_noisy_images("scd")
            self.denoise_and_save_samples(
                "scd", scd_noisy_images, fixed_timesteps, self.current_epoch
            )            
            print('scd success.')

