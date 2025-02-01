import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import dataset
from config.config import LDMTrainingConfig
# from models.AutoencoderKL_clf2_tanh import VAE
# from models.AutoencoderKL_clf2 import VAE
# from models.AutoencoderKL_clf2_sigmoid import VAE
from models.AutoencoderKL_clf2_sigmoid_adaptive_clf_new import VAE
# from models.AutoencoderKL import VAE
# from models.AutoencoderKL_condition_2 import VAE
# from models.AutoencoderKL_sigmoid import VAE
# from models.AutoencoderKL_clf2_tanh import VAE

from utils import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_vae_model(checkpoint_path, config):
    """Load a VAE model from a checkpoint."""
    print("Loading VAE model from:", checkpoint_path)
    vae = VAE.load_from_checkpoint(checkpoint_path, config=config)
    vae.to(device)
    vae.eval()
    return vae


def prepare_images(fname, key, indices):
    """Load and preprocess a batch of images."""
    dataset_obj = dataset.Dataset(
        fname_h5=os.path.join("/mydata/dlbirhoui/firat/OADAT", fname),
        key=key,
        transforms=transforms,
        inds=indices,
    )
    images = [
        torch.tensor(img, device=device, dtype=torch.float32).unsqueeze(0)
        for img in iter(dataset_obj)
    ]
    return torch.cat(images, dim=0)


def visualize_latent_space(latents, labels, output_path):
    """Visualize the latent space using UMAP and save the figure."""
    # Flatten latents to 2D array [n_samples, latent_dim]
    latents = latents.reshape(latents.shape[0], -1)

    # Perform UMAP dimensionality reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    latents_2d = umap_reducer.fit_transform(latents)

    # Plot the UMAP projection
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Domain Label")
    plt.title("UMAP Projection of Latent Space")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Save the figure
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Latent space visualization saved to: {output_path}")


if __name__ == "__main__":
    # Settings
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_tanh/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2/epoch=149-val_total_loss=0.0000.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_sigmoid/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_sigmoid_5000_fixed_lamda_dup/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_sigmoid_10000/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_sigmoid_new_5000/last.ckpt"
    vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_sigmoid_new_1000/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_lpips_disc/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_lpips_disc_clf_adapt_smaller_weight_0.5/epoch=211-val_total_loss=0.0000.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_before_sigmoid/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_after_sigmoid/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_tanh_5000_fixed_lamda/last.ckpt"
    # vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_tanh_5000/last.ckpt"

    output_dir = "./latent_space_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize config
    config = LDMTrainingConfig(batch_size=64)

    # Load VAE model
    vae = load_vae_model(vae_checkpoint_path, config=config)

    # Load predefined indices for SCD and SWFD
    indices_scd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_train_indices.npy")
    indices_swfd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/train_sc_BP_indices.npy")

    # Randomly pick 32 indices from each domain
    scd_indices = np.random.choice(indices_scd, 600, replace=False)
    swfd_indices = np.random.choice(indices_swfd, 600, replace=False)

    # Prepare SCD and SWFD domain images
    scd_fname_h5 = "SCD_RawBP.h5"
    scd_key = "vc_BP"
    scd_images = prepare_images(scd_fname_h5, scd_key, scd_indices)

    swfd_fname_h5 = "SWFD_semicircle_RawBP.h5"
    swfd_key = "sc_BP"
    swfd_images = prepare_images(swfd_fname_h5, swfd_key, swfd_indices)

    # Encode images into latent space
    with torch.no_grad():
        scd_latents = torch.sigmoid(vae.vae.encode(scd_images).latent_dist.sample()).cpu().numpy()
        swfd_latents = torch.sigmoid(vae.vae.encode(swfd_images).latent_dist.sample()).cpu().numpy()

    # Combine latents and labels
    latents = np.concatenate([scd_latents, swfd_latents], axis=0)
    labels = np.array([0] * len(scd_latents) + [1] * len(swfd_latents))  # 0 for SCD, 1 for SWFD

    # Visualize latent space
    output_path = os.path.join(output_dir, "latent_space_umap.png")
    visualize_latent_space(latents, labels, output_path)