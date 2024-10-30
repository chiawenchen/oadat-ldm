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
import torch.autograd as autograd

class DiffusionModel(LightningModule):

    def __init__(
        self,
        config: TrainingConfig,
        noise_scheduler: DDIMScheduler,
        classifier: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 32, 64, 64, 64, 128, 128),  # small model
            # block_out_channels=(64, 64, 128, 128, 256, 256), # medium model
            # block_out_channels=(128, 128, 256, 256, 512, 512), # large model
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
        self.classifier = classifier

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps)

    def save_image(
        self,
        image: torch.Tensor,
        save_locally: bool = False,
        local_path: str = None,
        log_to_wandb: bool = False,
        wandb_name: str = None,
        use_clip: bool = False,
    ) -> None:
        if use_clip:
            # check if the image has NaN or Inf values
            if torch.any(torch.isinf(image)) or torch.any(torch.isnan(image)):
                print(f"Image contains NaN or Inf values.")
            # check if the image is in the range [0, 1]
            if torch.min(image) < 0 or torch.max(image) > 1:
                print(
                    f"Image is not in the range [0, 1]. Min: {torch.min(image)}, Max: {torch.max(image)}"
                )
                # clip the image to [0, 1]
                image = torch.clamp(image, 0, 1)

        # Convert the tensor to a PIL image
        image = v2.ToPILImage()(image.cpu())

        # Save image locally if required
        if save_locally:
            if local_path is None:
                raise ValueError("local_path must be provided if save_locally is True")
            image.save(local_path)

        # Log image to WandB if required
        if log_to_wandb:
            if wandb_name is None:
                raise ValueError("wandb_name must be provided if log_to_wandb is True")
            self.logger.experiment.log({wandb_name: [wandb.Image(image)]})

    def save_images(
        self,
        images: list[torch.Tensor],
        timesteps: torch.Tensor,
        local_path: str = None,
        log_to_wandb: bool = False,
        wandb_name: str = None,
        use_clip: bool = False,
    ) -> None:
        if use_clip:
            # check if the image has NaN or Inf values
            for img in images:
                if torch.any(torch.isinf(img)) or torch.any(torch.isnan(img)):
                    print(f"Image contains NaN or Inf values.")
            # check if the image is in the range [0, 1]
            images = [torch.clamp(img, min=0, max=1) for img in images]

        num_images = len(images)
        num_rows = 2
        num_cols = math.ceil(num_images / num_rows)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        axs = axs.ravel()

        for i, img in enumerate(images):
            ax = axs[i] if num_images > 1 else axs
            im = ax.imshow(img.squeeze(0).cpu(), cmap="gray", aspect="equal")
            ax.axis("off")
            ax.set_title(f"t = {timesteps[i] + 1}")
            fig.colorbar(im, ax=ax)

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.savefig(local_path, bbox_inches="tight")
        plt.close()

        # Log image to WandB if required
        if log_to_wandb:
            if wandb_name is None:
                raise ValueError("wandb_name must be provided if log_to_wandb is True")
            self.logger.experiment.log({wandb_name: [wandb.Image(local_path)]})

    # Function to generate fixed noisy images before training
    def generate_fixed_noisy_images(self, category: str) -> None:
        if category == "val":
            clean_images = next(iter(self.trainer.datamodule.val_dataloader())).to(
                self.device
            )
        elif category == "scd":
            clean_images = next(iter(self.trainer.datamodule.scd_dataloader())).to(
                self.device
            )

        clean_image = clean_images[0]  # Use the first image only

        # save clean image
        self.save_image(
            clean_image,
            save_locally=True,
            local_path=os.path.join(self.config.sample_dir, f"{category}_clean.png"),
            log_to_wandb=True,
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

        torch.manual_seed(self.config.seed)
        noises = torch.randn(clean_image.shape, device=self.device)

        noisy_images = []
        for idx, t in enumerate(timesteps):
            noisy_image = self.noise_scheduler.add_noise(clean_image, noises, t)
            noisy_image_path = os.path.join(
                self.config.fixed_image_paths[category],
                f"{category}_noisy_image_timestep_{t.item()}.pt",
            )
            torch.save(noisy_image, noisy_image_path)
            noisy_images.append(noisy_image)

        # save noisy images
        local_path = os.path.join(self.config.sample_dir, f"{category}_noisy.png")
        self.save_images(
            noisy_images,
            timesteps,
            local_path,
            log_to_wandb=True,
            wandb_name=f"{category}_noisy",
        )

        torch.seed()  # Reset the seed

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
        self.generate_fixed_noisy_images("val")
        self.generate_fixed_noisy_images("scd")

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
        # torch.manual_seed(self.config.seed)
        clean_images = batch
        noises = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=self.device,
            dtype=torch.int64,
        )

        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        noise_preds = self(noisy_images, timesteps).sample
        val_loss = self.loss_fn(noise_preds, noises)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def get_fixed_noisy_images(
        self, category: str, timesteps: torch.Tensor
    ) -> torch.Tensor:
        file_paths = [
            os.path.join(
                self.config.fixed_image_paths[category],
                f"{category}_noisy_image_timestep_{t.item()}.pt",
            )
            for t in timesteps
        ]

        # Load all noisy images and stack them into a single tensor
        noisy_images = torch.stack(
            [torch.load(path, weights_only=True).to(self.device) for path in file_paths]
        )

        return noisy_images

    # classifier-guided gradient step
    # def classifier_guided_step(self, noisy_image, timestep, guidance_scale=1.0):
    #     # Predict the noise as usual
    #     noise_pred = self.model(noisy_image, timestep).sample

    #     # Enable gradients for the classifier guidance step
    #     noisy_image.requires_grad_(True)
    #     with torch.enable_grad():
    #         # Use the classifier to get the probabilities
    #         classifier_logits = self.classifier(noisy_image)
    #         target_class = torch.ones_like(classifier_logits[:, 1])  # Targeting class 1 (swfd_obj)
    #         classifier_loss = F.cross_entropy(classifier_logits, target_class)

    #         # Backpropagate to get the gradient of the loss w.r.t. the noisy image
    #         grads = autograd.grad(outputs=classifier_loss, inputs=noisy_image)[0]

    #     # Adjust the predicted noise using the classifier gradient
    #     guided_noise_pred = noise_pred - guidance_scale * grads
    #     return guided_noise_pred

    # Predicts denoised images, and saves them in WandB and locally

    # classifier-guided denoising
    # def denoise_and_save_samples(
    #     self,
    #     category: str,
    #     noisy_images: torch.Tensor,
    #     timesteps: torch.Tensor,
    #     epoch: int,
    #     guidance_scale=1.0
    # ) -> None:
    #     self.noise_scheduler.set_timesteps(
    #         num_inference_steps=self.noise_scheduler.config.num_train_timesteps
    #     )

    #     denoised_images = []
    #     for i, noisy_image in enumerate(noisy_images):
    #         start_timestep = timesteps[i].item()  # Get the specific timestep for this image
    #         for t in reversed(range(start_timestep + 1)):  # Start from specific timestep, go down to 0
    #             timestep = torch.tensor([t], device=self.device)
    #             # Perform classifier-guided denoising
    #             guided_noise_pred = self.classifier_guided_step(
    #                 noisy_image.unsqueeze(0), timestep, guidance_scale
    #             )
    #             noisy_image = self.noise_scheduler.step(
    #                 guided_noise_pred.unsqueeze(0), # unsqueeze or not?
    #                 timestep.unsqueeze(0),
    #                 noisy_image.unsqueeze(0),
    #             ).images[0]

    #         # Append the final denoised image after reaching timestep 0
    #         denoised_images.append(noisy_image)

    #     # Save the denoised images
    #     local_path = os.path.join(
    #         self.config.sample_dir, f"{category}_epoch_{epoch}_denoised.png"
    #     )
    #     self.save_images(
    #         denoised_images,
    #         timesteps,
    #         local_path,
    #         log_to_wandb=True,
    #         wandb_name=f"{category} denoised",
    #         use_clip=True,
    #     )

    def denoise_and_save_samples(
        self,
        category: str,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        epoch: int,
    ) -> None:
        noise_preds = self(noisy_images, timesteps).sample
        self.noise_scheduler.set_timesteps(
            num_inference_steps=self.noise_scheduler.config.num_train_timesteps
        )
        # denoise the images
        denoised_images = [
            self.noise_scheduler.step(
                noise_preds[i], timesteps[i], noisy_images[i]
            ).pred_original_sample
            for i in range(len(noisy_images))
        ]
        # save the denoised images
        local_path = os.path.join(
            self.config.sample_dir, f"{category}_epoch_{epoch}_denoised.png"
        )
        self.save_images(
            denoised_images,
            timesteps,
            local_path,
            log_to_wandb=True,
            wandb_name=f"{category} denoised",
            use_clip=True,
        )

    def on_validation_epoch_end(self) -> None:
        if (
            self.current_epoch % self.config.save_image_epochs == 0
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
            val_noisy_images = self.get_fixed_noisy_images("val", fixed_timesteps)
            self.denoise_and_save_samples(
                "val", val_noisy_images, fixed_timesteps, self.current_epoch
            )

            # 2. Use the fixed noisy images for the SCD set
            scd_noisy_images = self.get_fixed_noisy_images("scd", fixed_timesteps)
            self.denoise_and_save_samples(
                "scd", scd_noisy_images, fixed_timesteps, self.current_epoch
            )

    def predict_step(self, image: torch.Tensor, timestep: torch.Tensor):
        self.eval()
        image = image.to(self.device)

        # define noise
        noise = torch.randn(image.shape).to(self.device)

        # add noise to an image
        self.noise_scheduler.set_timesteps(num_inference_steps=self.config.num_inference_steps)
        noisy_image = self.noise_scheduler.add_noise(image, noise, timestep)

        # predict noise
        noise_pred = self(noisy_image, timestep).sample

        # denoise using the model
        denoised_image = self.noise_scheduler.step(
            noise_pred, timestep, noisy_image
        ).pred_original_sample
        return denoised_image