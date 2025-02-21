import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import dataset
from models.CVAE_after_sigmoid import CVAE
from utils import transforms, load_config_from_yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_vae_model(checkpoint_path, config):
    """Load a VAE model from a checkpoint."""
    print("Loading VAE model from:", checkpoint_path)
    vae = CVAE.load_from_checkpoint(checkpoint_path, config=config)
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

def decode_latents(denoised_latents, labels, vae):
    """ Decode latents into images using the CVAE decoder. """

    # Project labels into the same latent space
    label_embeds = vae.label_embedding(labels)  # (B, latent_channels)

    # Compute scale & shift for conditioning
    scale = vae.label_scale(label_embeds).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
    shift = vae.label_shift(label_embeds).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
    
    # Apply feature-wise affine transformation
    conditioned_latents = denoised_latents * scale + shift  # Broadcasting (B, C, H, W)

    # Decode latents to reconstruct images
    decoded_images = vae.vae.decode(conditioned_latents).sample  # CVAE decoder
    return decoded_images

if __name__ == "__main__":
    vae_checkpoint_path = "/mydata/dlbirhoui/chia/checkpoints/all/cvae-after-5000/last.ckpt"

    output_dir = "./assets/latent_space_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize config
    config = load_config_from_yaml("/mydata/dlbirhoui/chia/oadat-ldm/config/cvae_after_5000.yaml")

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

    # labels
    swfd_labels = torch.ones(swfd_latents.shape[0], dtype=int, device=device)
    scd_labels = torch.ones(scd_latents.shape[0], dtype=int, device=device)
    labels = torch.stack([swfd_labels, scd_labels], dim=0)

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
    labels_dup = []

    for noise_level in noise_levels:
        perturbed_latents.append(latents + noise * noise_level)
        labels_dup.append(labels)
    
    perturbed_latents = torch.stack(perturbed_latents, dim=0)  # Shape: (num of levels, batch_size, latent_dim)
    labels_dup = torch.stack(labels_dup, dim=0)
    # Decode perturbed latents
    perturbed_latents = perturbed_latents.view(-1, *perturbed_latents.shape[2:])
    labels_dup = labels_dup.view(-1, *labels_dup.shape[2:])

    with torch.no_grad():
        decoded_images = decode_latents(perturbed_latents, labels_dup,vae)
        # decoded_images = vae.vae.decode(perturbed_latents).sample.cpu()
    
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
    
    plt.savefig(os.path.join(output_dir, "latent_perturbation_grid_cvae_after_5000.png"))
    plt.close()