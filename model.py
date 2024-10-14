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
        self.example_input_array = (torch.randn(self.config.batch_size, 1, self.config.image_size, self.config.image_size), torch.randint(0, self.config.num_train_timesteps, (self.config.batch_size,)))

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
 
