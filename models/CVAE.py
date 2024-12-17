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

class CVAE(LightningModule):
    def __init__(self, sample_dir, in_channels=1, latent_dim=512, img_size=256, block_out_channels=(64, 128, 256, 512), num_classes=2):
        super().__init__()
        self.sample_dir = sample_dir
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, block_out_channels[0], kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(block_out_channels[0], block_out_channels[1], kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(block_out_channels[1], block_out_channels[2], kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(block_out_channels[2], block_out_channels[3], kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(block_out_channels[3], block_out_channels[3], kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(block_out_channels[3], block_out_channels[3], kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
        )
        
        # Latent layers
        feature_dim = block_out_channels[-1] * (img_size // 64) ** 2  # 8x6 spatial dimensions # 64 for 4 x 4
        self.mean = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)
        
        # Latent to feature map
        self.latent_to_features = nn.Linear(latent_dim + num_classes, feature_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(block_out_channels[-1], block_out_channels[-1], kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(block_out_channels[-1], block_out_channels[-2], kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(block_out_channels[-2], block_out_channels[-3], kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(block_out_channels[-3], block_out_channels[-4], kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(block_out_channels[-4], block_out_channels[-4] // 2, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(block_out_channels[-4] // 2, in_channels, kernel_size=4, stride=2, padding=1),  # 256x256
        )
    
    def encode(self, x, labels):
        """
        Encodes the input into a latent distribution.
        """
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()  # Convert labels to one-hot
        one_hot_labels = one_hot_labels.view(-1, self.num_classes, 1, 1)  # Expand dimensions
        one_hot_labels = one_hot_labels.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, one_hot_labels], dim=1)

        features = self.encoder(x).flatten(start_dim=1)  # Flatten spatial dimensions
        mean = self.mean(features)
        logvar = self.logvar(features)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """
        Samples from the latent distribution using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, labels):
        """
        Decodes the latent representation into an image.
        """
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        z = torch.cat([z, one_hot_labels], dim=1)
        features = self.latent_to_features(z).view(-1, 512, 4, 4)  # Reshape to feature map
        reconstruction = self.decoder(features)
        return reconstruction
    
    def forward(self, x, labels):
        """
        Full forward pass through the VAE.
        """
        mean, logvar = self.encode(x, labels)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z, labels)
        return reconstructed, mean, logvar

    def compute_loss(self, reconstructed, original, mean, logvar):
        """
        Computes the VAE loss: reconstruction + KL divergence.
        """
        recon_loss = F.mse_loss(reconstructed, original, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss /= original.numel()  # Normalize by the number of elements
        return recon_loss + 0.001 * kl_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        """
        Training step: compute loss and log metrics.
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        reconstructed, mean, logvar = self.forward(images, labels)
        loss, recon_loss, kl_loss = self.compute_loss(reconstructed, images, mean, logvar)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train_kl_loss", kl_loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: compute loss and log metrics.
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        reconstructed, mean, logvar = self.forward(images, labels)
        loss, recon_loss, kl_loss = self.compute_loss(reconstructed, images, mean, logvar)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True)
        
        return loss

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
            swfd_images, swfd_labels = [], []
            scd_images, scd_labels = [], []

            for idx, batch in enumerate(val_iterator):
                images, labels = batch
                if idx < 5:  # First 5 batches
                    swfd_images.append(images[0].unsqueeze(0))
                    swfd_labels.append(labels[0].unsqueeze(0))
                if idx >= len(val_dataloader) - 5:  # Last 5 batches
                    scd_images.append(images[0].unsqueeze(0))
                    scd_labels.append(labels[0].unsqueeze(0))
            
            # Combine images
            swfd_images = torch.cat(swfd_images, dim=0)
            swfd_labels = torch.cat(swfd_labels, dim=0)
            scd_images = torch.cat(scd_images, dim=0)
            scd_labels = torch.cat(scd_labels, dim=0)
            originals = torch.cat([swfd_images, scd_images], dim=0).to(self.device)
            labels = torch.cat([swfd_labels, scd_labels], dim=0).to(self.device)

            # Generate reconstructions
            reconstructed, _, _ = self.forward(originals, labels)
            
            # Plot and save the figure
            file_path = f"{self.sample_dir}/validation_epoch_{self.current_epoch}.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })

            # Swap the labels
            swapped_labels = torch.cat([scd_labels, swfd_labels], dim=0).to(self.device)

            # Generate reconstructions
            mean, logvar = self.encode(originals, labels)
            z = self.reparameterize(mean, logvar)
            reconstructed = self.decode(z, swapped_labels)
            
            # Plot and save the figure
            file_path = f"{self.sample_dir}/validation_epoch_{self.current_epoch}_swap.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Encode with Original Labels, Decode with Swapped Labels": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })

            # randomly sample 10 noises
            num_samples = 10
            random_latents = torch.randn(num_samples, self.latent_dim).to(self.device)
            random_labels = torch.cat([
                torch.zeros(5, dtype=torch.long),  # First 5 to class 0
                torch.ones(5, dtype=torch.long)   # Last 5 to class 1
            ]).to(self.device)

            # generate images by decoding the noise
            generated_images = self.decode(random_latents, random_labels)

            # Plot and save the figure
            gen_file_path = f"{self.sample_dir}/generated_epoch_{self.current_epoch}.png"
            fig, axes = plt.subplots(1, num_samples, figsize=(30, 5))
            for i in range(num_samples):
                ax = axes[i]
                img = generated_images[i].squeeze(0).cpu().detach().numpy()  # Convert to NumPy
                im = ax.imshow(img, cmap='gray')
                ax.axis('off')
                ax.set_title(f"Class {random_labels[i].item()}", fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(gen_file_path)
            plt.close(fig)
            
            # Log the generated images to WandB
            self.logger.experiment.log({
                "Generated Images": wandb.Image(gen_file_path, caption=f"Epoch {self.current_epoch}: Generated Images")
            })