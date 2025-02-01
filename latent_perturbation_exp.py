import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import dataset
from config.config import LDMTrainingConfig
from models.AutoencoderKL_clf2_sigmoid import VAE
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


if __name__ == "__main__":
    vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/vae/aekl_clf2_sigmoid/last.ckpt"

    output_dir = "./latent_space_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize config
    config = LDMTrainingConfig(batch_size=64)

    # Load VAE model
    vae = load_vae_model(vae_checkpoint_path, config=config)

    # Load predefined indices for SCD and SWFD
    indices_scd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_test_indices.npy")
    indices_swfd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/test_sc_BP_indices.npy")

    # Randomly pick 32 indices from each domain
    num_sample = 8
    scd_indices = np.random.choice(indices_scd, num_sample, replace=False)
    swfd_indices = np.random.choice(indices_swfd, num_sample, replace=False)
    print('scd indices: ', scd_indices)
    print('swfd indices: ', swfd_indices)

    # Prepare SCD and SWFD domain images
    scd_fname_h5 = "SCD_RawBP.h5"
    scd_key = "vc_BP"
    scd_images = prepare_images(scd_fname_h5, scd_key, scd_indices)

    swfd_fname_h5 = "SWFD_semicircle_RawBP.h5"
    swfd_key = "sc_BP"
    swfd_images = prepare_images(swfd_fname_h5, swfd_key, swfd_indices)

    # Encode images into latent space
    with torch.no_grad():
        scd_latents = torch.sigmoid(vae.vae.encode(scd_images).latent_dist.sample())
        swfd_latents = torch.sigmoid(vae.vae.encode(swfd_images).latent_dist.sample())

    # Combine latents and labels
    latents = torch.stack([scd_latents, swfd_latents], dim=0) # Shape: (batch_size, latent_dim)
    latents = latents.reshape(-1, config.latent_channels, config.latent_size, config.latent_size)
    print("latents' shape: ", latents.shape)

    # Create fixed noise
    local_rng = torch.Generator(device=device).manual_seed(66)
    noise = torch.randn((1, config.latent_size, config.latent_size), device=device, generator=local_rng)

    # Perturbate latents with different levels of noise
    noise_levels = torch.linspace(0, 0.1, 11)  # 1% to 10%
    perturbed_latents = []

    for noise_level in noise_levels:
        perturbed_latents.append(latents + noise * noise_level)
    
    perturbed_latents = torch.stack(perturbed_latents, dim=0)  # Shape: (num of levels, batch_size, latent_dim)

    # Decode perturbed latents
    perturbed_latents = perturbed_latents.view(-1, *perturbed_latents.shape[2:])

    with torch.no_grad():
        decoded_images = vae.vae.decode(perturbed_latents).sample.cpu()
    
    # Visualize decoded latents: 10 by 16 grid
    fig, axes = plt.subplots(noise_levels.shape[0], 16, figsize=(38, 25), constrained_layout=True)
    fig.suptitle("Latent Perturbation Visualization - After Decoding", fontsize=18)
    decoded_images = decoded_images.numpy()

    for i in range(noise_levels.shape[0]):  # Noise levels
        axes[i, 0].set_ylabel(f"Noise {i}%", fontsize=10)
        for j in range(16):  # Samples
            img = decoded_images[i * 16 + j].squeeze()
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].spines['bottom'].set_visible(False)
            axes[i, j].spines['left'].set_visible(False)
    
    for j in range(num_sample):
        axes[0, j].set_title(f"SCD {j+1}", fontsize=8)

    for idx, j in enumerate(range(num_sample, 2*num_sample)):
        axes[0, j].set_title(f"SWFD {idx+1}", fontsize=8)
    
    plt.savefig(os.path.join(output_dir, "latent_perturbation_grid.png"))
    plt.close()