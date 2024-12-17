# vqvae
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
from diffusers import VQModel

class VAE(LightningModule):
    def __init__(self, config):
        super().__init__()

        down_block_types = tuple("DownEncoderBlock2D" for _ in range(config.num_down_blocks))
        up_block_types = tuple("UpDecoderBlock2D" for _ in range(config.num_up_blocks))
        block_out_channels = tuple(config.block_out_channels)

        self.vqvae = VQModel(
            in_channels=1,
            out_channels=1,
            block_out_channels=block_out_channels,
            num_vq_embeddings=config.num_vq_embeddings,
            vq_embed_dim=config.vq_embed_dim,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            latent_channels=config.latent_channels,
        )
        self.config = config

    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for the VQ-VAE. Encodes the input, quantizes it, and reconstructs it.
        """
        encoded_outputs = self.vqvae.encode(x)
        latents = encoded_outputs.latent  # Quantized latents
        reconstructed = self.vqvae.decode(latents)
        return reconstructed, latents

    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images = batch.to(self.device)

        # Forward pass
        outputs = self.vqvae(images)
        reconstructed = self.vqvae.decode(latents)
        quantization_loss = outputs.loss  # Contains codebook and commitment losses

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, images, reduction="mean")

        # Total loss
        loss = recon_loss + quantization_loss

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train_quant_loss", quantization_loss, on_step=False, on_epoch=True)

        return loss

    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        images = batch.to(self.device)

        # Forward pass
        outputs = self.vqvae(images)
        reconstructed = outputs.reconstruction
        quantization_loss = outputs.loss

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, images, reduction="mean")

        # Total loss
        loss = recon_loss + quantization_loss

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_quant_loss", quantization_loss, on_step=False, on_epoch=True)

        return loss


    def configure_optimizers(self):
        """
        Optimizer configuration.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = LinearLR(
            optimizer, total_iters=self.config.lr_warmup_epochs, last_epoch=-1
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
        if self.current_epoch % self.config.save_image_epochs == 0:
            val_dataloader = self.trainer.datamodule.val_dataloader()
            val_iterator = iter(val_dataloader)
            images = []

            # Collect images
            for idx, batch in enumerate(val_iterator):
                if idx < 10:  # Collect first 10 images
                    images.append(batch.unsqueeze(0))

            images = torch.cat(images, dim=0).to(self.device)

            # Generate reconstructions
            reconstructed, _ = self.forward(images)

            # Plot and save the figure
            file_path = f"{self.config.sample_dir}/validation_epoch_{self.current_epoch}.png"
            fig = self.plot(images, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)

            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed": wandb.Image(
                    file_path,
                    caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)"
                )
            })

            # Randomly sample latents
            num_samples = 5
            random_indices = torch.randint(
                0, self.vqvae.num_vq_embeddings, (num_samples, self.config.latent_channels, self.config.latent_size, self.config.latent_size)
            ).to(self.device)
            quantized_latents = self.vqvae.codebook.embedding(random_indices)
            quantized_latents = quantized_latents.permute(0, 3, 1, 2)  # Rearrange to [B, C, H, W]

            # Generate images by decoding the quantized latents
            generated_images = self.vqvae.decode(quantized_latents)

            # Plot and save the generated images
            gen_file_path = f"{self.config.sample_dir}/generated_epoch_{self.current_epoch}.png"
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
            for i in range(num_samples):
                ax = axes[i]
                img = generated_images[i].squeeze(0).cpu().detach().numpy()
                im = ax.imshow(img, cmap='gray')
                ax.axis('off')
                ax.set_title(f"Sample {i+1}", fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(gen_file_path)
            plt.close(fig)

            # Log the generated images to WandB
            self.logger.experiment.log({
                "Generated Images": wandb.Image(gen_file_path, caption=f"Epoch {self.current_epoch}: Generated Images")
            })
