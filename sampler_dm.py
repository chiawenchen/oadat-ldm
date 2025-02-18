import os
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from diffusers import DDIMScheduler
import dataset
from models.DDIM import DiffusionModel
from models.UnetClassifier import UnetAttentionClassifier
from utils import (
    get_last_checkpoint,
    get_named_beta_schedule,
    transforms,
    load_config_from_yaml,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_model(
    checkpoint_path, model_class, config=None, noise_scheduler=None, **kwargs
):
    """Load model from checkpoint."""
    print("loading a model from: ", checkpoint_path)

    if issubclass(model_class, DiffusionModel):
        model = model_class.load_from_checkpoint(
            checkpoint_path, config=config, noise_scheduler=noise_scheduler
        )
    elif issubclass(model_class, UnetAttentionClassifier):
        model = model_class.load_from_checkpoint(checkpoint_path, config)
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


def plot_batch_results(images, indices, output_dir, filename, category, titles=None):
    """
    Plot and save the results for each image in the batch.
    If `titles` is provided (a list of strings), they are used to annotate the subplots.
    """
    # num_images = len(images)
    # fig, axs = plt.subplots(1, num_images, figsize=(4 * num_images, 8))
    num_cols = 10
    fig, axs = plt.subplots(len(images) // num_cols, num_cols, figsize=(4 * num_cols, 8 * len(images) // num_cols))
    axs = axs.ravel()
    if len(images) == 1:
        axs = [axs]

    # Preprocess images if necessary (e.g., for display)
    if category != "noisy":
        images = [
            torch.clamp((((img + 1.0) * 1.2 / 2.0) - 0.2), min=-0.2, max=1.0)
            for img in images
        ]

    images = [img.squeeze(0).detach().cpu().numpy() for img in images]

    for i, ax in enumerate(axs):
        if category == "noisy":
            im = ax.imshow(images[i], cmap="gray")
        else:
            im = ax.imshow(images[i], cmap="gray", vmin=-0.2, vmax=1.0)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout(w_pad=0.25, h_pad=0.25)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def classifier_gradient(x, t, y, classifier, scale):
    """Compute the classifier gradient in chunks to handle large batches."""
    gradients = []
    chunk_size = max(1, len(x) // 2)  # adjust chunk size as needed

    with torch.enable_grad():
        for i in range(0, len(x), chunk_size):
            x_chunk = x[i : i + chunk_size]
            t_chunk = t[i : i + chunk_size]
            y_chunk = y[i : i + chunk_size]

            x_chunk.requires_grad_(True)
            logits = classifier(x_chunk, t_chunk)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y_chunk.view(-1)]
            grad_chunk = torch.autograd.grad(
                selected.sum(), x_chunk, retain_graph=False
            )[0]
            gradients.append(grad_chunk)
            x_chunk.requires_grad_(False)

    return torch.cat(gradients, dim=0) * scale


def sampling(
    model,
    images,
    timestep,
    classifier=None,
    classifier_scale=10.0,
    target_class=1,
    prediction_type="v_prediction",
    guidance_start=None,
    guidance_stop=None,
):
    """
    Perform diffusion sampling (denoising) for a given number of timesteps.

    If classifier guidance is provided, then guidance is applied only when:
         - guidance_start and guidance_stop are not None, and
         - the current timestep t is in [guidance_stop, guidance_start].
    If guidance_start/stop are not provided, guidance is applied at all timesteps.
    """
    denoised_images = images

    # If classifier guidance is enabled, create the target labels.
    if classifier is not None:
        y = torch.full(
            (images.size(0),), target_class, dtype=torch.long, device=images.device
        )

    for t in tqdm(range(timestep, -1, -1), desc="Sampling"):
        t_tensor = torch.full(
            (denoised_images.size(0),), t, dtype=torch.int64, device=images.device
        )
        pred = model(denoised_images, t).sample

        # Decide whether to apply classifier guidance:
        apply_guidance = False
        if classifier is not None:
            if (guidance_start is not None) and (guidance_stop is not None):
                if guidance_start >= t >= guidance_stop:
                    apply_guidance = True
            else:
                apply_guidance = True

        if apply_guidance:
            classifier_grad = classifier_gradient(
                denoised_images, t_tensor, y, classifier, classifier_scale
            )
            alpha_prod_t = model.noise_scheduler.alphas_cumprod[t]
            alpha_prod_t_sqrt = torch.sqrt(alpha_prod_t)
            beta_prod_t_sqrt = torch.sqrt(1 - alpha_prod_t)
            if prediction_type == "epsilon":
                pred = pred - beta_prod_t_sqrt * classifier_grad
            elif prediction_type == "v_prediction":
                # Convert v-prediction to epsilon, adjust, and convert back.
                pred_epsilon = (
                    alpha_prod_t_sqrt * pred + beta_prod_t_sqrt * denoised_images
                )
                pred_epsilon = pred_epsilon - beta_prod_t_sqrt * classifier_grad
                pred = (
                    pred_epsilon - beta_prod_t_sqrt * denoised_images
                ) / alpha_prod_t_sqrt
            else:
                print("Not defined prediction type: ", prediction_type)

        with torch.no_grad():
            denoised_images = [
                model.noise_scheduler.step(
                    pred[i], t_tensor[i], denoised_images[i]
                ).prev_sample
                for i in range(len(denoised_images))
            ]
            denoised_images = torch.stack(denoised_images)

    return denoised_images


def generate_filename(model_name, extra_info):
    """Generate an adjustable output filename."""
    return f"{model_name}_{extra_info}.png"


# Main Execution
if __name__ == "__main__":
    # --- Fixed Settings ---
    num_train_steps, num_inference_steps = 1000, 1000
    output_dir = "./assets/report"
    config = load_config_from_yaml(
        "/mydata/dlbirhoui/chia/oadat-ldm/config/diffusion-model.yaml"
        # "/mydata/dlbirhoui/chia/oadat-ldm/config/diffusion-model-epsilon.yaml"
    )
    clf_config = load_config_from_yaml(
        "/mydata/dlbirhoui/chia/oadat-ldm/config/classifier-guidance.yaml"
    )
    model_name = "dm"  # config.wandb.job_name

    # Create a noise scheduler for the diffusion model.
    noise_scheduler = get_named_beta_schedule(config.noise_schedule, num_train_steps)
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Load the diffusion model.
    dm_ckpt = os.path.join(
        config.paths.output_dir,
        "checkpoints",
        "all",
        config.wandb.job_name,
        "last.ckpt",
    )
    diffusion_model = load_model(
        dm_ckpt, DiffusionModel, config=config, noise_scheduler=noise_scheduler
    )

    # --- Define Parameter Combinations ---
    # Each dictionary specifies one configuration for one output image.
    # Keys:
    #   seed: random seed for noise generation
    #   use_classifier_guidance: whether to use classifier guidance
    #   classifier_scale: scale for the classifier gradient
    #   forward_timestep: number of timesteps used to add noise (the forward process)
    #   backward_timestep: total number of denoising steps (the backward process)
    #   guidance_start/guidance_stop: only apply guidance for backward steps in this range
    #   two_stage: if True and backward_timestep > forward_timestep, apply two-stage denoising.
    # Find the best timestep for sampling
    param_combinations = [
        {
            "seed": config.seed,
            "use_classifier_guidance": False,
            "classifier_scale": 0.0,
            "forward_timestep": step,
            "backward_timestep": step,
            "guidance_start": None,
            "guidance_stop": None,
            "two_stage": False,
        } for step in range(990, 1000, 1)
    ]


    # If any combination requires classifier guidance, load the classifier once.
    if any(combo["use_classifier_guidance"] for combo in param_combinations):
        classifier_model = load_model(
            os.path.join(clf_config.ckpt_dir, "last.ckpt"), UnetAttentionClassifier
        )
    else:
        classifier_model = None

    # --- Prepare the Base Image ---
    # Here we either sample from a dataset or use pure noise.
    sample_from_pure_noise = False
    if sample_from_pure_noise:
        # Pure noise: create a dummy image (its content will be replaced by noise)
        base_image = torch.zeros(
            (1, 1, config.image_size, config.image_size), device=device
        )
    else:
        scd_fname_h5 = "SCD_RawBP.h5"
        scd_key = "vc_BP"
        batch_indices = np.array([19960])
        base_image = prepare_images(scd_fname_h5, scd_key, batch_indices)

    # --- Process Each Parameter Combination ---
    images_row = []  # to store one output image per combination
    titles = []  # descriptive title for each output image

    for combo in param_combinations:
        current_seed = combo["seed"]
        current_forward = combo["forward_timestep"]
        current_backward = combo["backward_timestep"]
        two_stage = combo["two_stage"]
        guidance_start = combo["guidance_start"]
        guidance_stop = combo["guidance_stop"]

        # Decide on classifier usage for this combination.
        current_classifier = (
            classifier_model if combo["use_classifier_guidance"] else None
        )

        # Create a noise generator with the given seed.
        rng = torch.Generator(device=device).manual_seed(current_seed)
        noise = torch.randn(base_image.shape, device=device, generator=rng)

        # Add noise using the specified forward timestep.
        if current_forward < 0:
            noisy_image = base_image
        else:
            noisy_image = diffusion_model.noise_scheduler.add_noise(
                base_image,
                noise,
                torch.full(
                    (base_image.size(0),),
                    current_forward,
                    dtype=torch.int64,
                    device=device,
                ),
            )

        # --- Denoising (Backward Process) ---
        # If two_stage is enabled and extra steps are required, perform two-stage denoising.
        if two_stage and (current_backward > current_forward):
            # Stage 1: denoise from forward timestep down to 0.
            stage1 = sampling(
                diffusion_model,
                noisy_image,
                current_forward,
                classifier=current_classifier,
                classifier_scale=combo["classifier_scale"],
                prediction_type=diffusion_model.noise_scheduler.config.prediction_type,
                guidance_start=guidance_start,
                guidance_stop=guidance_stop,
            )
            extra_steps = current_backward - current_forward
            # Stage 2: assume the image is at a later timestep (extra_steps) and denoise further.
            # (This assumes that the model/scheduler can be used in this way.)
            final_image = sampling(
                diffusion_model,
                stage1,
                extra_steps,
                classifier=current_classifier,
                classifier_scale=combo["classifier_scale"],
                prediction_type=diffusion_model.noise_scheduler.config.prediction_type,
                guidance_start=guidance_start,
                guidance_stop=guidance_stop,
            )
        else:
            # Single-stage denoising from current_backward down to 0.
            final_image = sampling(
                diffusion_model,
                noisy_image,
                current_backward,
                classifier=current_classifier,
                classifier_scale=combo["classifier_scale"],
                prediction_type=diffusion_model.noise_scheduler.config.prediction_type,
                guidance_start=guidance_start,
                guidance_stop=guidance_stop,
            )

        # Save the first (or only) image from the batch.
        images_row.append(final_image[0])
        title_str = (
            f"Seed: {current_seed}\n"
            f"Guidance: {'On' if combo['use_classifier_guidance'] else 'Off'}\n"
            f"Scale: {combo['classifier_scale']}\n"
            f"F: {current_forward + 1}, B: {current_backward + 1}\n"
            f"Guidance [{guidance_start + 1}, {guidance_stop + 1}]\n"
            f"2-Stage: {'On' if two_stage else 'Off'}"
        )
        titles.append(title_str)

    # --- Plot the Combined Results ---
    filename = generate_filename(model_name, f"scd={str(batch_indices[0])}_vpred")
    plot_batch_results(
        images_row,
        list(range(len(images_row))),
        output_dir,
        filename,
        "denoised",
        titles=titles,
    )
    print(f"Saved combined result to {os.path.join(output_dir, filename)}")
