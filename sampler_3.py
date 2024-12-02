import os
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import TrainingConfig
from diffusers import DDIMScheduler
from utils import (
    get_last_checkpoint,
    get_named_beta_schedule,
    scd_transforms,
)
from model_one_more_layer import DiffusionModel
# from model_more_attention import DiffusionModel
from train_classifier import ResNetClassifier
from UnetClassifier import UnetClassifier
from UnetAttentionClassifier import UnetAttentionClassifier
import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_model(checkpoint_path, model_class, config, noise_scheduler):
    """Load model from checkpoint."""
    print(checkpoint_path)
    model = model_class.load_from_checkpoint(checkpoint_path, config=config, noise_scheduler=noise_scheduler)
    model.to(device)
    model.eval()
    return model


def prepare_images(data_path, fname, key, indices, transforms):
    """Load and preprocess a batch of images."""
    dataset_obj = dataset.Dataset(
        fname_h5=os.path.join(data_path, fname),
        key=key,
        transforms=transforms,
        inds=indices,
    )
    images = [
        torch.tensor(img, device=device, dtype=torch.float32).unsqueeze(0)
        for img in dataset_obj
    ]
    return torch.cat(images, dim=0)

def plot_batch_results(images, indices, output_dir, path_prev, category):
    """Plot and save the results for each image in the batch."""
    # Create a 4x4 grid for the 16 images
    fig, axs = plt.subplots(4, 4, figsize=(28, 28))

    # Preprocess images based on category
    if category == "denoised":
        images = [
            # torch.clamp((img - img.min()) / (img.max() - img.min()), min=0, max=1)
            (((img + 1.0) * 1.2 / 2.0) - 0.2)
            for img in images
        ]
    images = [img.squeeze(0).detach().cpu().numpy() for img in images]
    images = images[:16]
    indices = indices[:16]

    axs = axs.flatten()

    for ax, img, idx in zip(axs, images, indices):
        im = ax.imshow(img, cmap="gray", vmin=-0.2, vmax=1)
        ax.set_title(f"scd_idx={idx}")
        fig.colorbar(im, ax=ax)
        ax.axis("off")

    fig.suptitle(f"{category}", fontsize=24)

    # Save the figure
    output_path = os.path.join(output_dir, f"{path_prev}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_results(originals, noisies, denoised_batch, indices, output_dir, path_prev):
    """Plot and save the results for each image in the batch."""
    for i, (original, noisy, denoised, idx) in enumerate(
        zip(originals, noisies, denoised_batch, indices)
    ):
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))
        denoised = (denoised + 1) / 2.0
        images = [
            original.transpose(1, 2, 0),
            denoised.squeeze(0).detach().cpu().numpy(),
            noisy.squeeze(0).detach().cpu().numpy(),
        ]
        titles = ["Original Image", "Denoised Image", "Noisy Image"]

        for ax, img, title in zip(axs, images, titles):
            im = ax.imshow(img, cmap="gray", vmin=-0.2, vmax=1.0)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)

        fig.suptitle(f"Diffusion Sampling (scd_idx={idx})", fontsize=16)
        output_path = os.path.join(output_dir, f"{path_prev}_scd_id={idx}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

def classifier_gradient(x, t, y, classifier, scale):
    """Compute the classifier gradient in chunks to handle large batches."""
    gradients = []
    chunk_size = len(x) // 2

    with torch.enable_grad():
        for i in range(0, len(x), chunk_size):
            # Slice the batch into smaller chunks to fit into memory
            x_chunk = x[i : i + chunk_size]
            t_chunk = t[i : i + chunk_size]
            y_chunk = y[i : i + chunk_size]

            # Enable gradient calculation for the chunk
            x_chunk.requires_grad_(True)

            # Forward pass
            logits = classifier(x_chunk, t_chunk)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y_chunk.view(-1)]

            # Compute gradients
            grad_chunk = torch.autograd.grad(
                selected.sum(), x_chunk, retain_graph=False
            )[0]
            gradients.append(grad_chunk)

            # Disable gradient for the chunk
            x_chunk.requires_grad_(False)

    # Concatenate all gradients
    return torch.cat(gradients, dim=0) * scale


def sampling(
    model,
    images,
    timestep,
    original_images,
    batch_indices,
    output_dir,
    classifier=None,
    classifier_scale=1.0,
    target_class=1,
):
    """ Perform diffusion sampling with or without classifier guidance. """
    denoised_images = images

    # guide towards the target class: swfd
    if classifier is not None:
        y = torch.full(
            (images.size(0),), target_class, dtype=torch.long, device=images.device
        )

    for t in tqdm(range(timestep, -1, -1), desc="Sampling"):
        t_tensor = torch.full(
            (denoised_images.size(0),), t, dtype=torch.int64, device=images.device
        )
        noise_pred = model(denoised_images, t).sample

        if classifier is not None:
            # Compute classifier gradient and adjust noise prediction
            classifier_grad = classifier_gradient(
                denoised_images, t_tensor, y, classifier, classifier_scale
            )
            alpha_bar_sqrt = torch.sqrt(1 - model.noise_scheduler.alphas_cumprod[t])
            noise_pred = noise_pred - alpha_bar_sqrt * classifier_grad

        # Perform denoising step
        with torch.no_grad():
            denoised_images = [
                model.noise_scheduler.step(
                    noise_pred[i], t_tensor[i], denoised_images[i]
                ).prev_sample
                for i in range(len(denoised_images))
            ]
            denoised_images = torch.stack(denoised_images)

        # with torch.no_grad():
        #     if (t + 1) % 50 == 0 or t == 0:
        #         plot_results(original_images.cpu().numpy(), images, denoised_images, batch_indices, output_dir, f"t={t+1}")

        # Clear unused memory
        torch.cuda.empty_cache()

    return denoised_images


def preprocess_labels(images):
    # replace gray pixels into white
    non_black_mask = images > 0
    images[non_black_mask] = 1.0
    return images


# Main Execution
if __name__ == "__main__":
    # settings
    num_sampling = 16
    forward_timestep = 995
    backward_timestep = 995
    use_classifier_guidance = False
    num_inference_steps = 1000
    num_train_steps = 1000
    noise_scheduler = get_named_beta_schedule("cosine", num_train_steps)

    print("use_classifier_guidance: ", use_classifier_guidance)

    # Load models
    diffusion_model = load_model(
        # '/mydata/dlbirhoui/chia/checkpoints/ddim_small_mean0_deep_cosine_variety/epoch=249-val_loss=0.0080.ckpt',
        # "/mydata/dlbirhoui/chia/checkpoints/ddim_small_mean0_mix_deep_cosine_variety/epoch=248-val_loss=0.0057.ckpt",
        # "/mydata/dlbirhoui/chia/checkpoints/dm/epoch=210-val_loss=0.0100.ckpt",
        # "/mydata/dlbirhoui/chia/checkpoints/dm_mix/epoch=223-val_loss=0.0075.ckpt",
        # "/mydata/dlbirhoui/chia/checkpoints/dm_shift_atten/epoch=210-val_loss=0.0179.ckpt",
        "/mydata/dlbirhoui/chia/checkpoints/dm_shift/epoch=210-val_loss=0.0179.ckpt",
        # "/mydata/dlbirhoui/chia/checkpoints/dm_shift_10warmup/epoch=210-val_loss=0.0179.ckpt",
        # "/mydata/dlbirhoui/chia/checkpoints/dm_shift_dark/last.ckpt",
        # "/mydata/dlbirhoui/chia/checkpoints/dm_shift_dark_rescale/epoch=23-val_loss=0.0198.ckpt",
        DiffusionModel,
        config=TrainingConfig(num_epochs=0, batch_size=1),
        noise_scheduler=noise_scheduler,
    )
    print('check the noise scheduler.')
    print('prediction_type: ', diffusion_model.noise_scheduler.config.prediction_type)
    print('timestep_spacing: ', diffusion_model.noise_scheduler.config.timestep_spacing)
    print('beta_schedule: ', diffusion_model.noise_scheduler.config.beta_schedule)

    if use_classifier_guidance:
        # classifier = load_model('../checkpoints/classifier/resnet_classifier/last.ckpt', ResNetClassifier)
        # classifier = load_model('/mydata/dlbirhoui/chia/checkpoints/classifier/unet_classifier_cosine/epoch=172-val_loss=0.0759.ckpt', UnetClassifier)
        # classifier = load_model(
        #     "/mydata/dlbirhoui/chia/checkpoints/classifier/attention_unet_classifier_cosine/epoch=124-val_loss=0.0703.ckpt",
        #     UnetAttentionClassifier,
        # )
        classifier = load_model(
            "/mydata/dlbirhoui/chia/checkpoints/classifier/attclf_normalize/epoch=76-val_loss=0.1194.ckpt",
            UnetAttentionClassifier,
        )

    # Define batch of images to sample
    data_path = "/mydata/dlbirhoui/firat/OADAT"
    scd_fname_h5 = "SCD_RawBP.h5"  # "SWFD_semicircle_RawBP.h5"
    scd_key = "vc_BP"  # "sc_BP"
    batch_indices = np.random.randint(0, 20000, size=num_sampling)
    print(f"Sampling from {scd_fname_h5}, key={scd_key}, indices={batch_indices}")

    # Load batch of images
    scd_image_batch = prepare_images(
        data_path, scd_fname_h5, scd_key, batch_indices, scd_transforms
    )
    if scd_key == 'labels':
        scd_image_batch = preprocess_labels(scd_image_batch)

    # Initialize noise
    local_rng = torch.Generator(device="cuda")
    seed = 66
    local_rng.manual_seed(seed)
    noise = torch.randn(scd_image_batch.shape, device=device, generator=local_rng) # + 0.001 * torch.randn(scd_image_batch.shape[0], scd_image_batch.shape[1], 1, 1, device=device, generator=local_rng)
    noise_scheduler = diffusion_model.noise_scheduler
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    noisy_images = noise_scheduler.add_noise(
        scd_image_batch,
        noise,
        torch.full(
            (scd_image_batch.size(0),),
            forward_timestep,
            dtype=torch.int64,
            device=device,
        ),
    )

    output_dir = "./"

    if use_classifier_guidance:
        # Sampling with classifier guidance for the batch
        print("Running the classifier-guided sampling loop...")
        classifier_scale = 10
        denoised_images = sampling(
            diffusion_model,
            noisy_images,
            backward_timestep,
            scd_image_batch,
            batch_indices,
            output_dir,
            classifier,
            classifier_scale,
        )
        plot_batch_results(
            denoised_images,
            batch_indices,
            output_dir,
            f"mix_attclfnorm_f={forward_timestep}_b={backward_timestep}_s={classifier_scale}_seed={seed}",
            "denoised",
        )
    else:
        print("Running the sampling loop...")
        denoised_images = sampling(
            diffusion_model,
            noisy_images,
            backward_timestep,
            scd_image_batch,
            batch_indices,
            output_dir,
        )
        plot_batch_results(
            denoised_images,
            batch_indices,
            output_dir,
            f"swfd_shift_f={forward_timestep}_b={backward_timestep}_seed={seed}_denoised",
            "denoised", # "noisy", "original"
        )
        # plot_batch_results(
        #     scd_image_batch,
        #     batch_indices,
        #     output_dir,
        #     f"swfd_shift_f={forward_timestep}_b={backward_timestep}_seed={seed}_clean",
        #     "original",
        # )
