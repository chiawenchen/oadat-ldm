from diffusers import AutoencoderKL
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F

class VAE(LightningModule):
    def __init__(self):
        super().__init__()
        
        # Load pretrained VAE
        self.pretrained_vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        
        # Modify the first layer of the encoder
        self.modify_encoder_input_layer()
        
        # Modify the last layer of the decoder
        self.modify_decoder_output_layer()
        
        # Latent dimension (optional if needed elsewhere)
        self.latent_dim = 4  # Example: Latent channels for AutoencoderKL
    
    def modify_encoder_input_layer(self):
        """
        Modify the input layer of the encoder to accept 1-channel input.
        """
        # Get the original first layer
        original_conv = self.pretrained_vae.encoder.conv_in
        
        # Create a new convolutional layer with 1 input channel
        new_conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
        )
        
        # Initialize the new layer's weights by averaging the original weights
        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv.bias.data = original_conv.bias.data
        
        # Replace the original layer with the new one
        self.pretrained_vae.encoder.conv_in = new_conv
    
    def modify_decoder_output_layer(self):
        """
        Modify the output layer of the decoder to produce 1-channel output.
        """
        # Get the original last layer
        original_conv = self.pretrained_vae.decoder.conv_out
        
        # Create a new convolutional layer with 1 output channel
        new_conv = torch.nn.Conv2d(
            in_channels=original_conv.in_channels,
            out_channels=1,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
        )
        
        # Initialize the new layer's weights by selecting the first channel of the original weights
        new_conv.weight.data = original_conv.weight.data[:, 0:1, :, :]
        new_conv.bias.data = original_conv.bias.data[:1]
        
        # Replace the original layer with the new one
        self.pretrained_vae.decoder.conv_out = new_conv
    
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
