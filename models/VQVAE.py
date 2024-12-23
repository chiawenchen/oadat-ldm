import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import LinearLR
from matplotlib import pyplot as plt
import wandb


class VQVAE(LightningModule):
    def __init__(self, sample_dir='./', in_channels=1, img_size=256, block_out_channels=(64, 128, 256, 512),
                 codebook_size=512, embedding_dim=64):
        super().__init__()
        self.latent_size = 4
        self.sample_dir = sample_dir
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(block_out_channels[0], block_out_channels[1], kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(block_out_channels[1], block_out_channels[2], kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(block_out_channels[2], block_out_channels[3], kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(block_out_channels[3], block_out_channels[3], kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(block_out_channels[3], embedding_dim, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
        )

        # Codebook for vector quantization
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, block_out_channels[-1], kernel_size=4, stride=2, padding=1),  # 8x8
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

    def encode(self, x):
        """
        Encodes input into continuous latent representation.
        """
        z_e = self.encoder(x)
        z_e_flatten = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)  # (B*H*W, D)
        return z_e, z_e_flatten

    def vector_quantization(self, z_e_flatten):
        """
        Performs vector quantization by finding the closest embeddings in the codebook.
        """
        distances = (torch.sum(z_e_flatten ** 2, dim=1, keepdim=True) +
                     torch.sum(self.codebook.weight ** 2, dim=1) -
                     2 * torch.matmul(z_e_flatten, self.codebook.weight.t()))  # (B*H*W, K)
        indices = torch.argmin(distances, dim=1)  # Closest embedding index
        z_q_flatten = self.codebook(indices)  # Quantized embeddings
        return z_q_flatten, indices

    def decode(self, z_q):
        """
        Decodes the quantized latent representation into an image.
        """
        # Ensure z_q has shape (B, embedding_dim, H, W)
        if len(z_q.shape) == 4 and z_q.shape[1] != self.embedding_dim:
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        reconstruction = self.decoder(z_q)
        return reconstruction


    def forward(self, x):
        """
        Full forward pass: encode, quantize, and decode.
        """
        z_e, z_e_flatten = self.encode(x)
        z_q_flatten, indices = self.vector_quantization(z_e_flatten)
        z_q = z_q_flatten.view_as(z_e)  # Reshape back to (B, H, W, embedding_dim)
        
        reconstructed = self.decode(z_q)
        return reconstructed, z_e, z_q

    def compute_loss(self, reconstructed, original, z_e, z_q):
        """
        Computes the VQ-VAE loss: reconstruction + commitment loss.
        """
        recon_loss = F.mse_loss(reconstructed, original, reduction="mean")
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="mean")
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="mean")
        total_loss = recon_loss + commitment_loss + codebook_loss
        return total_loss, recon_loss, commitment_loss, codebook_loss

    def training_step(self, batch, batch_idx):
        """
        Training step: compute loss and log metrics.
        """
        images = batch.to(self.device)
        reconstructed, z_e, z_q = self.forward(images)
        loss, recon_loss, commitment_loss, codebook_loss = self.compute_loss(reconstructed, images, z_e, z_q)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train_commitment_loss", commitment_loss, on_step=False, on_epoch=True)
        self.log("train_codebook_loss", codebook_loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: compute loss and log metrics.
        """
        images = batch.to(self.device)
        reconstructed, z_e, z_q = self.forward(images)
        loss, recon_loss, commitment_loss, codebook_loss = self.compute_loss(reconstructed, images, z_e, z_q)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_commitment_loss", commitment_loss, on_step=False, on_epoch=True)
        self.log("val_codebook_loss", codebook_loss, on_step=False, on_epoch=True)
        
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
            reconstructed, _, _ = self.forward(originals)
            
            # Plot and save the figure
            file_path = f"{self.sample_dir}/validation_epoch_{self.current_epoch}.png"
            fig = self.plot(originals, reconstructed, n_images=10)
            fig.savefig(file_path)
            plt.close(fig)
            
            # Log the saved image to WandB
            self.logger.experiment.log({
                "Original vs Reconstructed": wandb.Image(file_path, caption=f"Epoch {self.current_epoch}: Originals (Top) vs Reconstructed (Bottom)")
            })

            # randomly sample 5 noises
            num_samples = 5
            random_latents = torch.randn(num_samples, self.embedding_dim, self.latent_size, self.latent_size).to(self.device)

            # generate images by decoding the noise
            generated_images = self.decode(random_latents)

            # Plot and save the figure
            gen_file_path = f"{self.sample_dir}/generated_epoch_{self.current_epoch}.png"
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