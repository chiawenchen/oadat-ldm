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

class VAE(LightningModule):
    def __init__(self):
        super().__init__() 
        self.vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            sample_size=256,
            block_out_channels=(16, 32, 64, 128),
            norm_num_groups=4,
            down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")
        )
    
   def forward(self, x: torch.Tensor):
        """
        Forward pass for the VAE. Encodes the input into latent space and reconstructs it.
        """
        latents = self.vae.encode(x).latent_dist.sample()
        reconstructed = self.vae.decode(latents).sample
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
        recon_loss = F.mse_loss(reconstructed, images)
        kl_loss = latents_dist.kl().mean()
        loss = recon_loss + 0.1 * kl_loss
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True)
        self.log("train_kl_loss", kl_loss, on_step=True, on_epoch=True)
        
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
        recon_loss = F.mse_loss(reconstructed, images)
        kl_loss = latents_dist.kl().mean()
        loss = recon_loss + 0.1 * kl_loss
        
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
        fig, axes = plt.subplots(2, n_images, figsize=(15, 5))
        
        # Loop through the selected images
        for i in range(n_images):
            # Original images
            ax = axes[0, i]
            img = originals[i].squeeze(0).cpu().numpy()  # Convert to NumPy for plotting
            im = ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title("Originals", fontsize=12)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            
            # Reconstructed images
            ax = axes[1, i]
            img = reconstructions[i].squeeze(0).cpu().numpy()  # Convert to NumPy for plotting
            im = ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title("Reconstructed", fontsize=12)          
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
            
            # Extract first 5 and last 5 images
            val_iterator = iter(val_dataloader)
            first_images = next(val_iterator)[:5]
            last_images = list(val_iterator)[-1][:5]
            
            # Combine images
            originals = torch.cat([first_images, last_images], dim=0).to(self.device)
            
            # Generate reconstructions
            reconstructed, _ = self.forward(originals)
            
            # Plot and save the figure
            file_path = f"./sample/vae/validation_epoch_{self.current_epoch}.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })