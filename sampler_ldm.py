import os
import torch
import math
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from diffusers import DDIMScheduler
import dataset
from models.LDM import LDM
from models.LDMWithCVAE import LDMWithCVAE
from models.VAE_after_sigmoid import VAE
from models.CVAE_after_sigmoid import CVAE
from utils import get_last_checkpoint, get_named_beta_schedule, transforms as pre_transforms, load_config_from_yaml
from config.parser import parse_arguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_model(checkpoint_path, model_class, config=None, noise_scheduler=None, vae=None):
    """Load model from checkpoint."""
    print("loading a model from: ", checkpoint_path)

    # Load model with the appropriate arguments
    if issubclass(model_class, VAE):
        model = model_class.load_from_checkpoint(checkpoint_path, config=config)
    elif issubclass(model_class, CVAE):
        model = model_class.load_from_checkpoint(checkpoint_path, config=config)
    elif issubclass(model_class, LDM):
        model = model_class.load_from_checkpoint(checkpoint_path, config=config, noise_scheduler=noise_scheduler, vae=vae)
    elif issubclass(model_class, LDMWithCVAE):
        model = model_class.load_from_checkpoint(checkpoint_path, config=config, noise_scheduler=noise_scheduler, vae=vae)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    model.to(device)
    model.eval()
    return model


def prepare_images(fname, key, indices, pre_transforms):
    """Load and preprocess a batch of images."""
    dataset_obj = dataset.Dataset(
        fname_h5=os.path.join("/mydata/dlbirhoui/firat/OADAT", fname),
        key=key,
        transforms=pre_transforms,
        inds=indices,
    )
    images = [torch.tensor(img, device=device, dtype=torch.float32).unsqueeze(0) for img in iter(dataset_obj)]
    return torch.cat(images, dim=0)


def plot_batch_results(images, indices, output_dir, filename, category):
    """Plot and save the results for each image in the batch."""
    # Create a grid
    num_row = 10
    num_col = 10
    fig, axs = plt.subplots(num_row, num_col, figsize=(4 * num_col, 4 * num_row))

    # Preprocess images based on category
    images = [torch.clamp((((img + 1.0) * 1.2 / 2.0) - 0.2), min=-0.2, max=1.0) for img in images]
    images = [img.squeeze(0).detach().cpu().numpy() for img in images]

    axs = axs.flatten()

    for ax, img, idx in zip(axs, images, indices):
        im = ax.imshow(img, cmap="gray", vmin=-0.2, vmax=1.0)
        # ax.set_title(f"SWFD_sc_idx={idx}")
        # fig.colorbar(im, ax=ax)
        ax.axis("off")

    # fig.suptitle(f"{category}", fontsize=24)

    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout(w_pad=0.25)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def sampling(
    model,
    images,
    timestep,
    target_class=1,
):
    """ Perform diffusion sampling """
    denoised_images = images.clone()

    for t in tqdm(range(timestep, -1, -1), desc="Sampling"):
        t_tensor = torch.full((denoised_images.size(0),), t, dtype=torch.int64, device=images.device)
        pred = model(denoised_images, t).sample

        # Perform denoising step
        with torch.no_grad():
            denoised_images = [
                model.noise_scheduler.step(pred[i], t_tensor[i], denoised_images[i]).prev_sample
                for i in range(len(denoised_images))
            ]
            denoised_images = torch.stack(denoised_images)

    return denoised_images

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

# Main Execution
if __name__ == "__main__":
    config = load_config_from_yaml(
        "/mydata/dlbirhoui/chia/oadat-ldm/config/latent-diffusion-model-cvae-after-5000-small.yaml"
    )
    vae_config = load_config_from_yaml(config.paths.vae_config_path)
    num_sampling = 100
    forward_timestep_list = [0]
    backward_timestep_list = [0]
    sample_from_cvae_only = False
    plot_results = ["denoised"] # "original"

    # Fixed settings
    file_prefix = config.wandb.job_name
    num_inference_steps, num_train_steps = 1000, 1000
    noise_scheduler = get_named_beta_schedule(config.noise_schedule, num_train_steps)
    output_dir = "./assets/report"
    ldm_ckpt = os.path.join(config.paths.output_dir, "checkpoints", "all", config.wandb.job_name, "last.ckpt")

    if vae_config.cvae:
        vae = load_model(config.paths.vae_ckpt_dir, CVAE, config=vae_config)
        diffusion_model = load_model(ldm_ckpt, LDMWithCVAE, config=config, noise_scheduler=noise_scheduler, vae=vae)
    else:
        vae = load_model(config.paths.vae_ckpt_dir, VAE, config=vae_config)
        diffusion_model = load_model(ldm_ckpt, LDM, config=config, noise_scheduler=noise_scheduler, vae=vae)

    # For debbugging
    print("prediction_type: ", diffusion_model.noise_scheduler.config.prediction_type)
    print("timestep_spacing: ", diffusion_model.noise_scheduler.config.timestep_spacing)
    print("rescale_betas_zero_snr: ", diffusion_model.noise_scheduler.config.rescale_betas_zero_snr)
    print("beta_schedule: ", diffusion_model.noise_scheduler.config.beta_schedule)

    # Initialize noise
    noise_size = config.latent_size
    local_rng = torch.Generator(device="cuda").manual_seed(config.seed)
    noise = torch.randn((num_sampling, 1, noise_size, noise_size), device=device, generator=local_rng)

    # Initialize noise scheduler
    noise_scheduler = diffusion_model.noise_scheduler
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Prepare images: uncomment to use SWFD data
    # scd_fname_h5 = "SWFD_semicircle_RawBP.h5" 
    # scd_key = "sc_BP"
    # swfd_train = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/train_sc_BP_indices.npy")
    # swfd_test = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/test_sc_BP_indices.npy")
    # batch_indices = np.random.choice(np.concatenate((swfd_train, swfd_test)), num_sampling, replace=False)

    # Prepare images: uncomment to use SCD data
    scd_fname_h5 = "SCD_RawBP.h5" 
    scd_key = "vc_BP" # "labels"
    # scd_small_test_ind = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_test_indices.npy")
    # scd_small_train_ind = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_train_indices.npy")
    # # Adjust batch_indices based on your needs
    # batch_indices = np.random.choice(np.concatenate((scd_small_test_ind, scd_small_train_ind)), num_sampling, replace=False)
    batch_indices = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_test_indices.npy")

    print(f"Sampled from {scd_fname_h5}, key={scd_key}") # , indices={batch_indices}

    if scd_key == "labels":
        # transform_mask = v2.Lambda(lambda x: (x > 0).astype(np.float32)) # convert skins and vessels to white
        transform_mask = v2.Lambda(lambda x: (x > 0) * 0.5)  # convert skins and vessels to gray
        pre_transforms = v2.Compose([transform_mask, pre_transforms])

    scd_image_batch = prepare_images(scd_fname_h5, scd_key, batch_indices, pre_transforms)

    # Encode images to get the latents
    input_images = torch.sigmoid(vae.vae.encode(scd_image_batch).latent_dist.sample())
    input_images = input_images * 2.0 - 1.0

    # Plot original images
    if "original" in plot_results:
        filename = "scd_original_25.png"
        plot_batch_results(scd_image_batch, batch_indices, output_dir, filename, "original")

    if sample_from_cvae_only and vae_config.cvae:
        print('Sample from cvae only conditioning on swfd label!')
        filename = f"cvae_swfd_label.png"
        target_labels = torch.ones(input_images.shape[0], dtype=int, device=device)
        output = decode_latents(input_images, target_labels, vae)
        plot_batch_results(output, batch_indices, output_dir, filename, "denoised")

    else:
        for forward_timestep in forward_timestep_list:
            for backward_timestep in backward_timestep_list:
                print("Adding noise...")
                noisy_images = noise_scheduler.add_noise(
                    input_images,
                    noise,
                    torch.full(
                        (input_images.size(0),),
                        forward_timestep,
                        dtype=torch.int64,
                        device=device,
                    ),
                )

                # Generate filename
                filename = f"{file_prefix}_f={forward_timestep}_b={backward_timestep}_denoised.png"

                # Sampling
                denoised_output = sampling(
                    diffusion_model,
                    noisy_images,
                    backward_timestep,
                )

                # Decode denoised output
                denoised_output = (denoised_output + 1.0) / 2.0 # denoised_output
                if vae_config.cvae:
                    target_labels = torch.ones(denoised_output.shape[0], dtype=int, device=device)
                    denoised_output = decode_latents(denoised_output, target_labels, vae)
                else:
                    denoised_output = vae.vae.decode(denoised_output).sample

                # Save results
                plot_batch_results(denoised_output, batch_indices, output_dir, filename, "denoised")
