import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import dataset
from models.VAE_after_sigmoid import VAE
from models.CVAE_after_sigmoid import CVAE
from config.parser import parse_arguments
from utils import transforms, load_config_from_yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_vae_model(checkpoint_path, config):
    """Load a VAE model from a checkpoint."""
    print("Loading VAE model from:", checkpoint_path)
    if config.cvae:
        vae = CVAE.load_from_checkpoint(checkpoint_path, config=config)
    else:
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
    label_mapping = {0: 'SCD', 1: 'SWFD'}
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)

    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=scatter.cmap(scatter.norm(l))) 
               for l in unique_labels]
    plt.legend(handles, [label_mapping[l] for l in unique_labels], title="Domain Label")


    plt.colorbar(scatter, label="Domain Label")
    plt.title("UMAP Projection of Latent Space")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Save the figure
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Latent space visualization saved to: {output_path}")


if __name__ == "__main__":
    # Initialize config
    args = parse_arguments()
    config = load_config_from_yaml(args.config_path)
    output_dir = "./assets/latent_space_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Load VAE model
    ckpt = os.path.join(config.paths.vae_ckpt_dir, "last.ckpt")
    vae = load_vae_model(ckpt, config=config)

    # Load predefined indices for SCD and SWFD
    indices_scd = np.load(config.dataset.scd_train_indices)
    indices_swfd = np.load(config.dataset.swfd_train_indices)

    # Randomly pick indices from each domain
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
    output_path = os.path.join(output_dir, f"umap_{config.wandb.job_name}.png")
    visualize_latent_space(latents, labels, output_path)