from diffusers import AutoencoderKL
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F

class VAE(LightningModule):
    def __init__(self):
        super().__init__()
        
        # Load pretrained VAE
        self.pretrained_vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        
        # Latent dimension (optional if needed elsewhere)
        self.latent_dim = 4  # Example: Latent channels for AutoencoderKL
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass for the VAE. Encodes the input into latent space and reconstructs it.
        """
        latents = self.pretrained_vae.encode(x).latent_dist.sample()
        reconstructed = self.pretrained_vae.decode(latents).sample
        return reconstructed, latents
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Training step for the VAE. Computes the reconstruction loss and KL divergence.
        """
        images = batch  # Assuming the batch is (images, labels)
        images = images.to(self.device)
        
        # Encode and decode
        latents_dist = self.pretrained_vae.encode(images).latent_dist
        latents = latents_dist.sample()
        reconstructed = self.pretrained_vae.decode(latents).sample
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, images, reduction="mean")
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + latents_dist.logvar - latents_dist.mean.pow(2) - latents_dist.logvar.exp()
        )
        kl_loss = kl_loss / images.numel()  # Normalize by the number of elements
        
        # Combined loss
        loss = recon_loss + 0.001 * kl_loss
        
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
        latents_dist = self.pretrained_vae.encode(images).latent_dist
        latents = latents_dist.sample()
        reconstructed = self.pretrained_vae.decode(latents).sample
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, images, reduction="mean")
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + latents_dist.logvar - latents_dist.mean.pow(2) - latents_dist.logvar.exp()
        )
        kl_loss = kl_loss / images.numel()  # Normalize by the number of elements
        
        # Combined loss
        loss = recon_loss + 0.001 * kl_loss
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        optimizer = torch.optim.AdamW(self.pretrained_vae.parameters(), lr=1e-4)
        return optimizer
