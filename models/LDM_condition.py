# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from lightning.pytorch import LightningModule
from diffusers import UNet2DConditionModel, DDIMScheduler
from torchvision.transforms import v2
from config.config import LDMTrainingConfig
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from models.VAE import VAE

class ConditionalLatentDiffusionModel(LightningModule):
    def __init__(self, config: LDMTrainingConfig, noise_scheduler: DDIMScheduler, vae: VAE) -> None:
        super().__init__()
        self.sample_size = config.image_size // 2 ** (config.num_down_blocks - 1)
        self.in_channels = config.latent_channels
        self.out_channels = self.in_channels
        self.cross_attention_dim = 256
        self.min_factor = -22
        self.max_factor = 22
        self.min_factor_fixed = {
            "swfd": -0.12,
            "scd": -0.12,
            "prepicked_noise": -0.12
        }
        self.max_factor_fixed = {
            "swfd": 0.047,
            "scd": 0.047,
            "prepicked_noise": 0.047
        }
        self.config = config
        self.vae = vae
        self.model = UNet2DConditionModel(
            sample_size=self.sample_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            layers_per_block=2,
            center_input_sample=True,
            block_out_channels=(
                64,
                128,
                192,
                256,
            ),
            down_block_types=(
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D"
            ),
            cross_attention_dim=self.cross_attention_dim,
            class_embeddings_concat=True,
            norm_num_groups=1,
            num_class_embeds=2,
            # time_embedding_act_fn='silu',
        )
        self.noise_scheduler = noise_scheduler
        self.loss_fn = F.mse_loss
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
            torch.ones(self.config.batch_size)
        )
        self.generator = None
        self.class_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.cross_attention_dim
        )

        # Freeze the VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

    def get_label_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(torch.int64)
        embeddings = self.class_embedding(labels)  # Shape: (batch_size, self.cross_attention_dim)
        embeddings = embeddings.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, self.cross_attention_dim)
        return embeddings

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(torch.int64)
        encoder_hidden_states = self.get_label_embeddings(labels)

        return self.model(x, timesteps, encoder_hidden_states, class_labels=labels)

    def save_clean_image(
        self,
        image: torch.Tensor,
        local_path: str,
        wandb_name: str,
    ) -> None:
        plt.figure(figsize=(6, 6))
        im = plt.imshow(image.squeeze(0).cpu().detach().numpy(), cmap="gray")
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
                torch.clamp(((torch.clamp(img, min=-1.0, max=1.0) + 1.0) * 1.2 / 2.0) - 0.2, min=-0.2, max=1)
                for img in images
            ]

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
                    img.squeeze(0).cpu().detach().numpy(),
                    cmap="gray",
                    aspect="equal",
                    vmin=-0.2,
                    vmax=1.0,
                )
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

    # Function to generate fixed noisy images before training
    def generate_fixed_noisy_images_for_variety_check(
        self,
        category: str,
        same_X0: bool = False,
        same_noise: bool = False,
        num_sampling: int = 5,
    ) -> None:        
        noises = torch.randn(
            (num_sampling, self.in_channels, self.sample_size, self.sample_size), generator=self.generator, device=self.device
        )

        for idx in range(len(noises)):
            fixed_image_dir = f"{self.config.sample_dir}/{category}_fixed_noisy_images"
            os.makedirs(fixed_image_dir, exist_ok=True)
            noisy_image_path = os.path.join(
                fixed_image_dir,
                f"{category}_noisy_image_{idx}.pt",
            )
            torch.save(noises[idx], noisy_image_path)

        # # combine noisy images and push to wandb
        # self.save_images(
        #     noises,
        #     timesteps=torch.full((noises.size(0),), 999, dtype=torch.int64, device=self.device),
        #     local_path=os.path.join(self.config.sample_dir, f"{category}_noisy.png"),
        #     wandb_name=f"{category}_noisy",
        # )

    def generate_fixed_noisy_images(self, category: str) -> None:
        local_path = os.path.join(self.config.sample_dir, f"{category}_noisy.png")
        # if os.path.exists(local_path):
        #     return
        if category == "swfd":
            clean_images, _ = next(iter(self.trainer.datamodule.val_dataloader()))
        else:
            clean_images, _ = next(iter(self.trainer.datamodule.scd_dataloader()))

        clean_image = clean_images[0].to(self.device)
        # save clean image
        self.save_clean_image(
            clean_image,
            local_path=os.path.join(self.config.sample_dir, f"{category}_clean.png"),
            wandb_name=f"{category}_clean",
        )

        reconstructed, _ = self.vae.forward(clean_image.unsqueeze(0))

        # save clean image
        self.save_clean_image(
            reconstructed.squeeze(0),
            local_path=os.path.join(self.config.sample_dir, f"{category}_clean_reconstructed.png"),
            wandb_name=f"{category}_clean_reconstructed",
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
        posterior = self.vae.vae.encode(clean_image.unsqueeze(0)).latent_dist
        z = posterior.sample()
        z_flat = z.flatten()
        self.min_factor_fixed[category] = z_flat.min()
        self.max_factor_fixed[category] = z_flat.max()
        print(category, " scaler min: ", self.min_factor_fixed[category])
        print(category, " scaler max: ", self.max_factor_fixed[category])
        print(f"----------After Encoding - Image Latent-----------")
        print('min: ', z_flat.min())
        print('q25: ', torch.quantile(z_flat, 0.25))
        print('q50: ', torch.quantile(z_flat, 0.50))
        print('q75: ', torch.quantile(z_flat, 0.75))
        print('max: ', z_flat.max())
        print('std: ', z_flat.std())
        print('mean: ', z_flat.mean())
        print(" ")
        z = (z - self.min_factor_fixed[category]) / (self.max_factor_fixed[category] - self.min_factor_fixed[category])
        noises = torch.randn(
            z.shape, generator=self.generator, device=self.device
        )


        noisy_latents = []
        for idx, t in enumerate(timesteps):
            # create noisy latents
            noisy_latent = self.noise_scheduler.add_noise(z, noises, t).squeeze(0)
            noisy_latents.append(noisy_latent)

            # save locally
            path = f"{self.config.sample_dir}/{category}_fixed_noisy_images"
            os.makedirs(path, exist_ok=True)
            noisy_latent_path = os.path.join(path, f"{category}_noisy_image_timestep_{t.item():03}.pt",)
            torch.save(noisy_latent, noisy_latent_path)

        # # combine noisy images and push to wandb
        # self.save_images(
        #     noisy_latents,
        #     timesteps,
        #     local_path,
        #     wandb_name=f"{category}_noisy",
        # )

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

    def on_train_start(self) -> None:
        print("on_train_start")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)
        self.generate_fixed_noisy_images("swfd")
        self.generate_fixed_noisy_images("scd")
        self.generate_fixed_noisy_images_for_variety_check(
            "prepicked_noise", False, True
        )

    def on_train_batch_start(self, batch: torch.Tensor, batch_idx: int):
        images, labels = batch
        posterior = self.vae.vae.encode(images).latent_dist
        latents = posterior.sample()
        self.min_factor = latents.flatten().min()
        self.max_factor = latents.flatten().max()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        bs = images.shape[0]
        posterior = self.vae.vae.encode(images).latent_dist
        latents = posterior.sample()
        latents = (latents - self.min_factor) / (self.max_factor - self.min_factor)

        noises = torch.randn(latents.shape, device=self.device)
        # generates a tensor of random integers (timesteps) ranging from 0 to num_train_timesteps - 1 for each image in the batch
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=self.device,
            dtype=torch.int64,
        )

        # By sampling random timesteps for each image, the model gradually learns to predict the noise at all stages of the diffusion process (from slightly noisy images to very noisy ones)
        noisy_latents = self.noise_scheduler.add_noise(latents, noises, timesteps)

        # Predict v instead of noise
        predicted_v = self(noisy_latents, timesteps, labels).sample

        # Convert ground truth noise (noises) to v
        alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        ground_truth_v = alpha_t * noises - sigma_t * latents

        # Compute loss
        loss = self.loss_fn(predicted_v, ground_truth_v)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # fixed seed during validation to ensure the noise is always the same
        images, labels = batch
        bs = images.shape[0]
        posterior = self.vae.vae.encode(images).latent_dist
        latents = posterior.sample()
        latents = (latents - self.min_factor) / (self.max_factor - self.min_factor)
        noises = torch.randn(
            latents.shape, generator=self.generator, device=self.device
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            dtype=torch.int64,
            generator=self.generator,
            device=self.device,
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noises, timesteps)
        # Predict v instead of noise
        predicted_v = self(noisy_latents, timesteps, labels).sample

        # Convert ground truth noise (noises) to v
        alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        ground_truth_v = alpha_t * noises - sigma_t * latents

        # Compute loss
        val_loss = self.loss_fn(predicted_v, ground_truth_v)
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
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        epoch: int,
    ) -> None:
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
                    torch.ones(active_latents.shape[0], device=self.device)
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

        denoised_latents = denoised_latents * ((self.max_factor_fixed[category] - self.min_factor_fixed[category])) + self.min_factor_fixed[category]
 
        # decode from latent space to image space
        print("latents after denoising and scaling back with min-max:")
        print(category, " scaler min: ", self.min_factor_fixed[category])
        print(category, " scaler max: ", self.max_factor_fixed[category])
        for idx, latent in enumerate(denoised_latents):
            print(f"----------After Denoising - Image: {idx}-----------")
            latent_flatten = latent.flatten()
            print('min: ', latent_flatten.min())
            print('q25: ', torch.quantile(latent_flatten, 0.25))
            print('q50: ', torch.quantile(latent_flatten, 0.50))
            print('q75: ', torch.quantile(latent_flatten, 0.75))
            print('max: ', latent_flatten.max())
            print('std: ', latent_flatten.std())
            print('mean: ', latent_flatten.mean())
            print(" ")
        denoised_images = self.vae.vae.decode(denoised_latents).sample

        # Save the denoised images
        self.save_images(
            denoised_images,
            timesteps,
            local_path=os.path.join(self.config.sample_dir, f"{category}_epoch_{epoch}_denoised.png"),
            wandb_name=f"{category} denoised",
            transform_back=True,
        )

    def on_validation_epoch_end(self) -> None:
        print("on_validation_epoch_end")
        if (
            self.current_epoch == 0
            or self.current_epoch % self.config.save_image_epochs == 0
            or self.current_epoch == self.config.num_epochs - 1
            # self.current_epoch % 1 == 0
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
            print('swfd success.')

            # 2. Use the fixed noisy images for the SCD set
            scd_noisy_images = self.get_fixed_noisy_images("scd")
            self.denoise_and_save_samples(
                "scd", scd_noisy_images, fixed_timesteps, self.current_epoch
            )            
            print('scd success.')


            # 3. samples from different noise, same X_0
            noisy_images = self.get_fixed_noisy_images("prepicked_noise")
            fixed_timesteps = torch.full(
                (noisy_images.size(0),), 999, dtype=torch.int64, device=self.device
            )
            self.denoise_and_save_samples(
                "prepicked_noise", noisy_images, fixed_timesteps, self.current_epoch
            )
            print('prepicked_noise success.')

