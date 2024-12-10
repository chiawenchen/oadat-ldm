import os
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from diffusers import DDIMScheduler
import dataset

from config.config import TrainingConfig, ClassifierConfig
from models.DDIM import DiffusionModel
from models.UnetClassifier import UnetAttentionClassifier
from utils import (
    get_last_checkpoint,
    get_named_beta_schedule,
    transforms,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_model(checkpoint_path, model_class, config=None, noise_scheduler=None, **dict):
    """Load model from checkpoint."""
    print("loading a model from: ", checkpoint_path)
    
    # Load model with the appropriate arguments
    if issubclass(model_class, DiffusionModel):
        model = model_class.load_from_checkpoint(
            checkpoint_path, config=config, noise_scheduler=noise_scheduler
        )
    elif issubclass(model_class, UnetAttentionClassifier):
        config = ClassifierConfig()
        model = model_class.load_from_checkpoint(checkpoint_path, config.__dict__)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    model.to(device)
    model.eval()
    return model



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

def plot_batch_results(images, indices, output_dir, filename, category):
    """Plot and save the results for each image in the batch."""
    # Create a 4x4 grid for the 16 images
    fig, axs = plt.subplots(4, 4, figsize=(28, 28))

    # Preprocess images based on category
    if category != "noisy":
        images = [
            torch.clamp((((img + 1.0) * 1.2 / 2.0) - 0.2), min=-0.2, max=1.0)
            for img in images
        ]

    images = [img.squeeze(0).detach().cpu().numpy() for img in images]
    images = images[:16]
    indices = indices[:16]

    axs = axs.flatten()

    for ax, img, idx in zip(axs, images, indices):
        if category == 'noisy':
            im = ax.imshow(img, cmap="gray")
        else:
            im = ax.imshow(img, cmap="gray", vmin=-0.2, vmax=1.0)
        ax.set_title(f"scd_idx={idx}")
        fig.colorbar(im, ax=ax)
        ax.axis("off")

    fig.suptitle(f"{category}", fontsize=24)

    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_single_image(originals, noisies, denoised_batch, indices, output_dir, path_prev):
    """Plot and save the results for each image in the batch."""
    for i, (original, noisy, denoised, idx) in enumerate(
        zip(originals, noisies, denoised_batch, indices)
    ):
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))
        denoised = torch.clamp((((denoised + 1.0) * 1.2 / 2.0) - 0.2), min=-0.2, max=1.0)
        original = torch.clamp((((denoised + 1.0) * 1.2 / 2.0) - 0.2), min=-0.2, max=1.0)
        images = [
            original.squeeze(0).detach().cpu().numpy(),
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
    # original_images,
    # batch_indices,
    # output_dir,
    classifier=None,
    classifier_scale=1.0,
    target_class=1,
    prediction_type='v_prediction'
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
        pred = model(denoised_images, t).sample

        if classifier is not None:
            # Compute classifier gradient and adjust noise prediction
            classifier_grad = classifier_gradient(
                denoised_images, t_tensor, y, classifier, classifier_scale
            )
            alpha_prod_t = model.noise_scheduler.alphas_cumprod[t]
            alpha_prod_t_sqrt = torch.sqrt(alpha_prod_t)
            beta_prod_t_sqrt = torch.sqrt(1 - alpha_prod_t)

            if prediction_type == 'epsilon':
                pred = pred - beta_prod_t_sqrt * classifier_grad
            elif prediction_type == 'v_prediction':
                # Calculate noise prediction
                pred_epsilon = alpha_prod_t_sqrt * pred + beta_prod_t_sqrt * denoised_images
                pred_epsilon = pred_epsilon - beta_prod_t_sqrt * classifier_grad
                pred = (pred_epsilon - beta_prod_t_sqrt * denoised_images) / alpha_prod_t_sqrt

            else:
                print('not defined prediction type: ', prediction_type)

        # Perform denoising step
        with torch.no_grad():
            denoised_images = [
                model.noise_scheduler.step(
                    pred[i], t_tensor[i], denoised_images[i]
                ).prev_sample
                for i in range(len(denoised_images))
            ]
            denoised_images = torch.stack(denoised_images)

        # with torch.no_grad():
        #     if (t + 1) % 50 == 0 or t == 0:
        #         plot_single_image(original_images, images, denoised_images, batch_indices, output_dir, f"t={t+1}")

        # Clear unused memory
        torch.cuda.empty_cache()

    return denoised_images

def generate_filename(model_name, forward_timestep, backward_timestep, seed, category):
    """Generate an adjustable output filename."""
    return f"{model_name}_s=10_f={forward_timestep}_b={backward_timestep}_seed={seed}_{category}.png"

# Main Execution
if __name__ == "__main__":
    # settings
    num_sampling = 16
    forward_timestep = 900
    backward_timestep = 900
    sample_from_pure_noise = False
    model_name = "cf_scd"
    plot_results = ['denoised', 'original', 'noisy']
    use_classifier_guidance = True
    classifier_scale = 10
    config = TrainingConfig(num_epochs=0, batch_size=1)
    seed = config.seed

    num_inference_steps = 1000
    num_train_steps = 1000
    noise_scheduler = get_named_beta_schedule("cosine_dark", num_train_steps)
    output_dir = "./"

    # Load models
    diffusion_model = load_model(
        "/mydata/dlbirhoui/chia/checkpoints/dm_shift_v_rescale/last.ckpt",
        DiffusionModel,
        config=config,
        noise_scheduler=noise_scheduler,
    )
    classifier = None
    if use_classifier_guidance:
        classifier = load_model(
            "/mydata/dlbirhoui/chia/checkpoints/classifier/clf_v_atten/last.ckpt",
            UnetAttentionClassifier,
        )
    
    # Check models
    print("use_classifier_guidance: ", use_classifier_guidance)
    print('prediction_type: ', diffusion_model.noise_scheduler.config.prediction_type)
    print('timestep_spacing: ', diffusion_model.noise_scheduler.config.timestep_spacing)
    print('rescale_betas_zero_snr: ', diffusion_model.noise_scheduler.config.rescale_betas_zero_snr)
    print('beta_schedule: ', diffusion_model.noise_scheduler.config.beta_schedule)

    # Initialize noise
    local_rng = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn((num_sampling, 1, config.image_size, config.image_size), device=device, generator=local_rng)
    noise_scheduler = diffusion_model.noise_scheduler
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Sample from pure noise
    if sample_from_pure_noise:
        noisy_images = noise
        batch_indices = ['pure_noise'] * num_sampling
        print(f"Sampled from pure noise")

    # Sample from dataset
    else:
        scd_fname_h5 = "SCD_RawBP.h5" 
        scd_key = "vc_BP" # "labels" 
        scd_small_test_ind = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_test_indices.npy")
        batch_indices = scd_small_test_ind[-num_sampling:]
        if scd_key == 'labels':
            transform_mask = v2.Lambda(lambda x: (x > 0).astype(np.float32)) # convert skins and vessels to white
            transforms = v2.Compose([transform_mask, transforms])

        scd_image_batch = prepare_images(
            scd_fname_h5, scd_key, batch_indices
        )
        print(f"Sampled from {scd_fname_h5}, key={scd_key}, indices={batch_indices}")
        print('Adding noise...')
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

    # Plot original and noisy images
    if 'original' in plot_results and sample_from_pure_noise == False:
        filename = generate_filename(model_name, forward_timestep, backward_timestep, seed, category="original")
        plot_batch_results(scd_image_batch, batch_indices, output_dir, filename, "original")

    if 'noisy' in plot_results:
        filename = generate_filename(model_name, forward_timestep, backward_timestep, seed, category="noisy")
        plot_batch_results(noisy_images, batch_indices, output_dir, filename, "noisy")

    if 'denoised' not in plot_results:
        print('skip sampling!')

    # Generate filename
    filename = generate_filename(
        model_name,
        forward_timestep,
        backward_timestep,
        seed,
        category="denoised",
    )

    # Sampling
    denoised_images = sampling(
        diffusion_model,
        noisy_images,
        backward_timestep,
        # scd_image_batch,
        # batch_indices,
        # output_dir,
        classifier=classifier,
        classifier_scale=classifier_scale,
        prediction_type='v_prediction'
    )

    # Save results
    plot_batch_results(denoised_images, batch_indices, output_dir, filename, "denoised")