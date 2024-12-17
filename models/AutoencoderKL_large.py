import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from matplotlib import pyplot as plt
import io
import os
import wandb
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import LinearLR
from diffusers import AutoencoderKL

class VAE(LightningModule):
    def __init__(self, config, sample_dir="./"):
        super().__init__() 
        self.vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            latent_channels=64,
            sample_size=config.image_size,
            block_out_channels=(16, 32, 32, 64, 64, 128),
            norm_num_groups=4,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")
        )
        self.sample_dir = sample_dir
        self.config = config
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for the VAE. Encodes the input into latent space and reconstructs it.
        """
        posterior = self.vae.encode(x)
        print("posterior shape: ", posterior.shape)
        latents = posterior.latent_dist.sample()
        print("sample shape: ", latents.shape)
        reconstructed = self.vae.decode(latents).sample
        print("reconstructed shape: ", reconstructed.shape)
        return reconstructed, latents
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Training step for the VAE. Computes the reconstruction loss and KL divergence.
        """
        images = batch  # Assuming the batch is (images, labels)
        images = images.to(self.device)
        
        # Encode and decode
        latents_dist = self.vae.encode(images).latent_dist
        latents = latents_dist.sample()
        reconstructed = self.vae.decode(latents).sample
        
        # Reconstruction loss + KL loss
        recon_loss = F.mse_loss(reconstructed, images, reduction="mean")
        kl_loss = latents_dist.kl().mean()
        loss = recon_loss + 0.001 * kl_loss
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train_kl_loss", kl_loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Validation step for the VAE. Logs reconstruction and KL divergence losses.
        """
        images = batch
        images = images.to(self.device)
        
        # Encode and decode
        latents_dist = self.vae.encode(images).latent_dist
        latents = latents_dist.sample()
        reconstructed = self.vae.decode(latents).sample
        
        # Reconstruction loss + KL loss
        recon_loss = F.mse_loss(reconstructed, images, reduction="mean")
        kl_loss = latents_dist.kl().mean()
        loss = recon_loss + 0.001 * kl_loss
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        Optimizer configuration.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        lr_scheduler = LinearLR(
            optimizer, total_iters=5, last_epoch=-1
        )
        return [optimizer], [lr_scheduler]

    def plot(self, originals, reconstructions, n_images=10):
        """
        Create a grid of original and reconstructed images with colorbars.
        """
        fig, axes = plt.subplots(2, n_images, figsize=(30, 10))
        
        # Loop through the selected images
        for i in range(n_images):
            # Original images
            ax = axes[0, i]
            img = originals[i].squeeze(0).cpu().numpy()  # Convert to NumPy for plotting
            im = ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title("Originals", fontsize=10)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            
            # Reconstructed images
            ax = axes[1, i]
            img = reconstructions[i].squeeze(0).cpu().numpy()  # Convert to NumPy for plotting
            im = ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title("Reconstructed", fontsize=10)          
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        return fig

    def on_validation_epoch_end(self):
        """
        Logs original and reconstructed images with color maps to WandB every 5 epochs.
        """
        # Only log every 5 epochs
        if self.current_epoch % 5 == 0:
            # Get the validation dataloader
            val_dataloader = self.trainer.datamodule.val_dataloader()
            
            val_iterator = iter(val_dataloader)
            swfd_images = []
            scd_images = []

            for idx, batch in enumerate(val_iterator):
                if idx < 5:  # First 5 batches
                    swfd_images.append(batch[0].unsqueeze(0))
                if idx >= len(val_dataloader) - 5:  # Last 5 batches
                    scd_images.append(batch[0].unsqueeze(0))
            
            # Combine images
            swfd_images = torch.cat(swfd_images, dim=0)
            scd_images = torch.cat(scd_images, dim=0)
            originals = torch.cat([swfd_images, scd_images], dim=0).to(self.device)
            
            # Generate reconstructions
            reconstructed, _ = self.forward(originals)
            
            # Plot and save the figure
            file_path = f"{self.sample_dir}/validation_epoch_{self.current_epoch}.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })

            # # randomly sample 5 noises
            # num_samples = 5
            # random_latents = torch.randn(num_samples, self.latent_dim).to(self.device)

            # # generate images by decoding the noise
            # generated_images = self.decode(random_latents)

            # # Plot and save the figure
            # gen_file_path = f"{self.sample_dir}/generated_epoch_{self.current_epoch}.png"
            # fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
            # for i in range(num_samples):
            #     ax = axes[i]
            #     img = generated_images[i].squeeze(0).cpu().detach().numpy()  # Convert to NumPy
            #     im = ax.imshow(img, cmap='gray')
            #     ax.axis('off')
            #     ax.set_title(f"Sample {i+1}", fontsize=10)
            #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # plt.tight_layout()
            # plt.savefig(gen_file_path)
            # plt.close(fig)
            
            # # Log the generated images to WandB
            # self.logger.experiment.log({
            #     "Generated Images": wandb.Image(gen_file_path, caption=f"Epoch {self.current_epoch}: Generated Images")
            # })