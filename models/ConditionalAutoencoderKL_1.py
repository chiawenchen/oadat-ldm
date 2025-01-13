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
from models.PretrainedResnetClassifier import Classifier

class VAE(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False

        down_block_types = tuple("DownEncoderBlock2D" for _ in range(config.num_down_blocks))
        up_block_types = tuple("UpDecoderBlock2D" for _ in range(config.num_up_blocks))
        block_out_channels = tuple(config.block_out_channels)
        
        self.label_dim = config.num_classes

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
        
        # Classifier for adversarial training
        latent_dim = config.latent_channels * config.latent_size * config.latent_size
        # self.classifier = Classifier(latent_dim=latent_dim, num_classes=2)
        # self.classifier_loss_fn = nn.CrossEntropyLoss()
        self.label_projection = nn.Linear(1, config.latent_channels)
        self.config = config

    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

    def get_last_layer_encoder(self):
        return self.vae.encoder.conv_out.weight

    # def calculate_classifier_weight(self, reconstruction_loss, classifier_loss):
    #     """
    #     Calculate adaptive weight for classifier loss based on gradient norms with respect to the encoder parameters.
    #     """

    #     # Compute gradients of reconstruction loss
    #     rec_grads = torch.autograd.grad(reconstruction_loss, self.get_last_layer_encoder(), retain_graph=True, allow_unused=True)
    #     rec_grad_norm = sum(torch.norm(g) for g in rec_grads if g is not None)

    #     # Compute gradients of classifier loss
    #     cls_grads = torch.autograd.grad(classifier_loss, self.get_last_layer_encoder(), retain_graph=True, allow_unused=True)
    #     cls_grad_norm = sum(torch.norm(g) for g in cls_grads if g is not None)

    #     # Ensure gradient norms are tensors
    #     rec_grad_norm = torch.tensor(rec_grad_norm, device=reconstruction_loss.device, dtype=torch.float32)
    #     cls_grad_norm = torch.tensor(cls_grad_norm, device=reconstruction_loss.device, dtype=torch.float32)

    #     # Log gradient norms
    #     self.log("train/rec_grad_norm", rec_grad_norm, prog_bar=True)
    #     self.log("train/cls_grad_norm", cls_grad_norm, prog_bar=True)

    #     # Compute adaptive weight
    #     weight = rec_grad_norm / (cls_grad_norm + 1e-8)  # Avoid division by zero
    #     return torch.clamp(weight, 0.0, 5000.0).detach()  # Clip to avoid instability


    # def forward(self, x: torch.Tensor):
    #     """
    #     Forward pass for the VAE. Encodes the input into latent space and reconstructs it.
    #     """
    #     latents = self.vae.encode(x).latent_dist.sample()
    #     reconstructed = self.vae.decode(latents).sample
    #     return reconstructed, latents

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass for the conditional VAE. Encodes the input and reconstructs it.
        """
        
        # Encode and decode
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.sample()
        labels = labels.float().unsqueeze(-1)
        projected_labels = self.label_projection(labels)
        projected_labels = projected_labels.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        projected_labels = projected_labels.expand(-1, -1, latents.size(2), latents.size(3))

        conditioned_latents = latents + projected_labels
        reconstructed = self.vae.decode(conditioned_latents).sample
        return reconstructed, latents, posterior
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Training step for the VAE. Computes the reconstruction loss and KL divergence.
        """
        # opt_ae, opt_disc = self.optimizers()
        opt_ae, opt_disc = self.optimizers()

        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Encode and decode
        reconstructions, latents, posterior = self.forward(images, labels)
        latents_flat = latents.view(latents.size(0), -1)

        # Phase 1: Update Autoencoder (reconstruction + adversarial)
        if self.global_step % 2 == 0:
            self.toggle_optimizer(opt_ae)

            # Compute loss including LPIPS
            aeloss, log_dict_ae = self.loss(images, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            # # Compute adversarial loss (no detach): allow gradients to flow to encoder
            # logits_adv = self.classifier(latents_flat)

            # # Minimize negative classifier loss:  Negative because encoder tries to fool
            # adversarial_loss = -self.classifier_loss_fn(logits_adv, labels)

            # # Compute adaptive classifier weight
            # adaptive_weight = self.calculate_classifier_weight(aeloss, adversarial_loss)

            # Combine losses
            total_loss = aeloss # + adaptive_weight * adversarial_loss
            self.manual_backward(total_loss)
            opt_ae.step()
            opt_ae.zero_grad()
            self.untoggle_optimizer(opt_ae)

            # Add total_loss to log_dict_ae
            log_dict_ae["train/total_loss"] = total_loss.detach()  
            # log_dict_ae["train/adversarial_loss"] = adversarial_loss
            # log_dict_ae["train/adaptive_classifier_weight"] = adaptive_weight

            self.log("aeloss", aeloss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
            # self.log("adversarial_loss", adversarial_loss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
            # self.log("adaptive_classifier_weight", adaptive_weight, prog_bar=True, logger=False, on_step=True, on_epoch=False)
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

        # # Phase 3: Update Classifier
        # else:
        #     self.toggle_optimizer(opt_cls)
        #      # Detach latents for classifier training
        #     logits_cls = self.classifier(latents_flat.detach())  # Detach here
        #     classifier_loss = self.classifier_loss_fn(logits_cls, labels)

        #     self.manual_backward(classifier_loss)
        #     opt_cls.step()
        #     opt_cls.zero_grad()
        #     self.untoggle_optimizer(opt_cls)

        #     self.log("train/classifier_loss", classifier_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        #     return classifier_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Encode and decode
        reconstructions, latents, posterior = self.forward(images, labels)

        # Compute discriminator loss
        discloss, log_dict_disc = self.loss(
            images, reconstructions, posterior, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val")



        # # Compute adaptive classifier weight
        # with torch.set_grad_enabled(True):
        # Compute LPIPS loss
        aeloss, log_dict_ae = self.loss(
            images, reconstructions, posterior, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
            
            # # Compute classifier loss
            # latents_flat = latents.view(latents.size(0), -1) 
            # logits = self.classifier(latents_flat)
            # classifier_loss = self.classifier_loss_fn(logits, labels)

            # # Combine for total validation loss
            # adversarial_loss = -classifier_loss

            # adaptive_weight = self.calculate_classifier_weight(aeloss, adversarial_loss)
        total_loss = aeloss # + adaptive_weight * adversarial_loss

        # Add total_loss to log_dict_ae
        log_dict_ae["val/total_loss"] = total_loss.detach()

        # Log validation metrics
        self.log("val/aeloss", aeloss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # self.log("val/adversarial_loss", adversarial_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # self.log("val/classifier_loss", classifier_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return total_loss

    def configure_optimizers(self):
        # Optimizer for the VAE (encoder + decoder)
        opt_ae = torch.optim.Adam(self.vae.parameters(), lr=self.config.learning_rate)

        # Optimizer for the discriminator
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=self.config.learning_rate)

        # # Optimizer for the classifier
        # opt_cls = torch.optim.Adam(self.classifier.parameters(), lr=self.config.learning_rate)


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
        if self.current_epoch % self.config.save_image_epochs == 0:
            # Get the validation dataloader
            val_dataloader = self.trainer.datamodule.val_dataloader()
            
            val_iterator = iter(val_dataloader)
            swfd_images = []
            scd_images = []
            swfd_labels = []
            scd_labels = []

            for idx, batch in enumerate(val_iterator):
                images, labels = batch
                image, label = images[0], labels[0]
                if idx < 5:  # First 5 batches: extract 5 uncontinuous swfd images
                    swfd_images.append(image.unsqueeze(0))
                    swfd_labels.append(label.unsqueeze(0))
                if idx >= len(val_dataloader) - 5:  # Last 5 batches: extract 5 uncontinuous scd images
                    scd_images.append(image.unsqueeze(0))
                    scd_labels.append(label.unsqueeze(0))

            # Combine images
            swfd_images = torch.cat(swfd_images, dim=0)
            scd_images = torch.cat(scd_images, dim=0)
            swfd_labels = torch.cat(swfd_labels, dim=0)
            scd_labels = torch.cat(scd_labels, dim=0)
            originals = torch.cat([swfd_images, scd_images], dim=0).to(self.device)
            original_labels = torch.cat([swfd_labels, scd_labels], dim=0).to(self.device)
            
            # Generate reconstructions
            reconstructed, _, _ = self.forward(originals, original_labels)
            
            # Plot and save the figure
            file_path = f"{self.config.sample_dir}/validation_epoch_{self.current_epoch}.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })

            # Revert the labels
            swapped_labels = torch.cat([scd_labels, swfd_labels], dim=0).to(self.device)
            reconstructed, _, _ = self.forward(originals, swapped_labels)

            file_path = f"{self.config.sample_dir}/validation_epoch_{self.current_epoch}_swapped.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed - Swapped Labels": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })

            # # randomly sample 5 noises
            # num_samples = 5
            # random_latents = torch.randn(num_samples, self.config.latent_channels, self.config.latent_size, self.config.latent_size).to(self.device)

            # # generate images by decoding the noise
            # generated_images = self.vae.decode(random_latents).sample

            # # Plot and save the figure
            # gen_file_path = f"{self.config.sample_dir}/generated_epoch_{self.current_epoch}.png"
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