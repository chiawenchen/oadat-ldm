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
from models.LatentDomainClassifier import Classifier

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
        
        # Domain Classifier
        self.classifier = Classifier(latent_channels=config.latent_channels)
        self.domain_clf_loss_fn = nn.CrossEntropyLoss()
        
        self.config = config
        self.total_steps = 5000
    
    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

    def get_last_layer_encoder(self):
        return self.vae.encoder.conv_out.weight

    def log_latent_statistics(self, z: torch.Tensor):
        z_flat = z.flatten()
        print("global step: ", self.global_step)
        print('min: ', z_flat.min().item())
        print('q25: ', torch.quantile(z_flat, 0.25).item())
        print('q50: ', torch.quantile(z_flat, 0.50).item())
        print('q75: ', torch.quantile(z_flat, 0.75).item())
        print('max: ', z_flat.max().item())
        print('- std: ', z_flat.std().item())
        print('- mean: ', z_flat.mean().item())

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the VAE. Encodes the input into latent space and reconstructs it.
        """
        latents = self.vae.encode(x).latent_dist.sample()
        latents = torch.sigmoid(latents)
        reconstructed = self.vae.decode(latents).sample
        return reconstructed, latents

    def on_train_start(self):
        dataset_size = len(self.trainer.datamodule.train_dataloader())
        batch_size = self.trainer.datamodule.batch_size
        max_epochs = self.trainer.max_epochs
        total_steps = (dataset_size // batch_size + int(dataset_size % batch_size > 0)) * max_epochs
        self.total_steps = total_steps

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Training step for the VAE. Computes the reconstruction loss and KL divergence.
        """
        opt_ae, opt_disc = self.optimizers()

        images, labels = batch
        
        # Encode and decode
        posterior = self.vae.encode(images).latent_dist
        latents_sample = posterior.sample()
        latents = torch.sigmoid(latents_sample)

        # print latent's statistics
        if self.global_step % 100 == 0:
            print("---- before sigmoid ----")
            self.log_latent_statistics(latents_sample)
            print("---- after sigmoid ----")
            self.log_latent_statistics(latents)

        reconstructions = self.vae.decode(latents).sample

        # Phase 1: Update Autoencoder (reconstruction + adversarial)
        if self.global_step % 2 == 0:
            self.toggle_optimizer(opt_ae)

            # Compute loss including LPIPS
            aeloss, log_dict_ae = self.loss(images, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

          
            # domain classifier
            domain_loss = 0
            if self.current_epoch >= 10:
                alpha = self.config.classifier_scale
                if alpha < 0:
                    p = float(self.global_step / self.total_steps)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                domain_output = self.classifier(latents_sample, alpha)
                domain_loss = self.domain_clf_loss_fn(domain_output, labels)

            total_loss = aeloss + self.config.classifier_weight * domain_loss

            self.manual_backward(total_loss)
            opt_ae.step()
            opt_ae.zero_grad()
            self.untoggle_optimizer(opt_ae)

            # Add total_loss to log_dict_ae
            log_dict_ae["train/total_loss"] = total_loss.detach()  
            log_dict_ae["train/domain_loss"] = domain_loss
           
            self.log("aeloss", aeloss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
            self.log("domain_loss", domain_loss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return total_loss

        # Phase 2: Update Discriminator
        elif self.global_step % 2 == 1:
            self.toggle_optimizer(opt_disc)
            discloss, log_dict_disc = self.loss(
                images, reconstructions, posterior, 1, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            self.manual_backward(discloss)
            opt_disc.step()
            opt_disc.zero_grad()
            self.untoggle_optimizer(opt_disc)

            self.log("discloss", discloss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # Encode and decode
        posterior = self.vae.encode(images).latent_dist
        latents_sample = posterior.sample()
        latents = torch.sigmoid(latents_sample)
        reconstructions = self.vae.decode(latents).sample

        # Compute LPIPS loss
        aeloss, log_dict_ae = self.loss(
            images, reconstructions, posterior, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )

        # Compute discriminator loss
        discloss, log_dict_disc = self.loss(
            images, reconstructions, posterior, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val")

        # domain classifier
        domain_loss = 0
        if self.current_epoch >= 10:
            alpha = self.config.classifier_scale
            if alpha < 0:
                p = float(self.global_step / self.total_steps)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_output = self.classifier(latents_sample, alpha)
            domain_loss = self.domain_clf_loss_fn(domain_output, labels)
        total_loss = aeloss + self.config.classifier_weight * domain_loss

        # Add total_loss to log_dict_ae
        log_dict_ae["val/total_loss"] = total_loss.detach()
        log_dict_ae["val/domain_loss"] = domain_loss

        # Log validation metrics
        self.log("val/aeloss", aeloss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        
        log_dict = {**log_dict_ae, **log_dict_disc}
        self.log_dict(log_dict)

        return total_loss

    def configure_optimizers(self):
        # Optimizer for the VAE (encoder + decoder)
        combined_params = list(self.vae.parameters()) + list(self.classifier.parameters())
        opt_ae = torch.optim.Adam(combined_params, lr=self.config.learning_rate)

        # Optimizer for the discriminator
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=self.config.learning_rate)

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
            im = ax.imshow(img, cmap='gray', vmin=-1.0, vmax=1.0)
            ax.axis('off')
            ax.set_title("Originals", fontsize=10)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            
            # Reconstructed images
            ax = axes[1, i]
            img = reconstructions[i].squeeze(0).cpu().numpy()  # Convert to NumPy for plotting
            im = ax.imshow(img, cmap='gray', vmin=-1.0, vmax=1.0)
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
                images, labels = batch
                image, label = images[0], labels[0]
                if idx < 5:  # First 5 batches: extract 5 uncontinuous swfd images
                    swfd_images.append(image.unsqueeze(0))
                if idx >= len(val_dataloader) - 5:  # Last 5 batches: extract 5 uncontinuous scd images
                    scd_images.append(image.unsqueeze(0))
            
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