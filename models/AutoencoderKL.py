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
from losses.LPIPSWithDiscriminator import LPIPSWithDiscriminator

class VAE(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False

        down_block_types = tuple("DownEncoderBlock2D" for _ in range(config.num_down_blocks))
        up_block_types = tuple("UpDecoderBlock2D" for _ in range(config.num_up_blocks))
        block_out_channels = tuple(config.block_out_channels)

        self.vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            latent_channels=config.latent_channels,
            sample_size=config.image_size,
            block_out_channels=block_out_channels,
            norm_num_groups=4,
            down_block_types=down_block_types,
            up_block_types=up_block_types
        )
        self.loss = LPIPSWithDiscriminator(disc_start=config.disc_start, kl_weight=config.kl_loss_weight, disc_weight=config.disc_weight)
        self.config = config
    
    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

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
        opt_ae, opt_disc = self.optimizers()
        images = batch.to(self.device)
        
        # Encode and decode
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()
        reconstructions = self.vae.decode(latents).sample
        
        if self.global_step % 2 == 0:
            # train encoder+decoder+logvar
            self.toggle_optimizer(opt_ae)
            aeloss, log_dict_ae = self.loss(images, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.manual_backward(aeloss)
            opt_ae.step()
            opt_ae.zero_grad()
            self.untoggle_optimizer(opt_ae)


            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if self.global_step % 2 == 1:
            # train the discriminator
            self.toggle_optimizer(opt_disc)
            discloss, log_dict_disc = self.loss(images, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.manual_backward(discloss)
            opt_disc.step()
            opt_disc.zero_grad()
            self.untoggle_optimizer(opt_disc)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
            
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Validation step for the VAE. Logs reconstruction and KL divergence losses.
        """
        images = batch.to(self.device)
        
        # Encode and decode
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()
        reconstructions = self.vae.decode(latents).sample
        
        aeloss, log_dict_ae = self.loss(images, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(images, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(self.vae.parameters(),
                                  lr=self.config.learning_rate)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.config.learning_rate)

        # lr_scheduler_ae = LinearLR(
        #     opt_ae, total_iters=self.config.lr_warmup_epochs, last_epoch=-1
        # )
        # lr_scheduler_disc = LinearLR(
        #     opt_disc, total_iters=self.config.lr_warmup_epochs, last_epoch=-1
        # )

        return [opt_ae, opt_disc], []

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
        if self.current_epoch % self.config.save_image_epochs == 0:
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
            file_path = f"{self.config.sample_dir}/validation_epoch_{self.current_epoch}.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })

            # randomly sample 5 noises
            num_samples = 5
            random_latents = torch.randn(num_samples, self.config.latent_channels, self.config.latent_size, self.config.latent_size).to(self.device)

            # generate images by decoding the noise
            generated_images = self.vae.decode(random_latents).sample

            # Plot and save the figure
            gen_file_path = f"{self.config.sample_dir}/generated_epoch_{self.current_epoch}.png"
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
            for i in range(num_samples):
                ax = axes[i]
                img = generated_images[i].squeeze(0).cpu().detach().numpy()  # Convert to NumPy
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