import os
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import TrainingConfig
from diffusers import DDIMScheduler
from utils import transforms, get_last_checkpoint
from model import DiffusionModel
# from train_classifier import ResNetClassifier
from UnetClassifier import UnetClassifier
import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def load_model(checkpoint_path, model_class):
    """Load model from checkpoint."""
    print(checkpoint_path)
    model = model_class.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def prepare_images(data_path, fname, key, indices):
    """Load and preprocess a batch of images."""
    dataset_obj = dataset.Dataset(
        fname_h5=os.path.join(data_path, fname), key=key, transforms=transforms, inds=indices
    )
    images = [torch.tensor(img, device=device, dtype=torch.float32).unsqueeze(0) for img in dataset_obj]
    return torch.cat(images, dim=0)


def classifier_gradient(x, t, y, classifier, scale=1.0):
    """Compute the classifier gradient."""
    x.requires_grad_(True)
    logits = classifier(x, t)
    log_probs = F.log_softmax(logits, dim=-1)
    print("log_probs: ", log_probs.tolist()[0])
    selected = log_probs[range(len(logits)), y.view(-1)]
    grad = torch.autograd.grad(selected.sum(), x)[0]
    return grad * scale


def classifier_guided_sampling(model, classifier, images, timestep, target_class=1, classifier_scale=1.0):
    """Perform classifier-guided diffusion sampling for a batch of images."""
    denoised_images = images
    y = torch.full((images.size(0),), target_class, dtype=torch.long, device=device)

    for t in tqdm(range(timestep, -1, -1), desc="Sampling with Classifier Guidance"):
        t_tensor = torch.full((denoised_images.size(0),), t, dtype=torch.int64, device=device)
        noise_pred = model(denoised_images, t).sample
        classifier_grad = classifier_gradient(denoised_images, t_tensor, y, classifier, scale=classifier_scale)
        
        # Adjust noise prediction based on classifier gradient
        alpha_bar_sqrt = torch.sqrt(1 - model.noise_scheduler.alphas_cumprod[t])
        noise_pred = noise_pred - alpha_bar_sqrt * classifier_grad

        with torch.no_grad():
            denoised_images = [
                model.noise_scheduler.step(noise_pred[i], t_tensor[i], denoised_images[i]).prev_sample
                for i in range(len(denoised_images))
            ]
            denoised_images = torch.stack(denoised_images)

        torch.cuda.empty_cache()

    return denoised_images

def sampling(model, images, t):
    t_tensor = torch.full((images.size(0),), t, dtype=torch.int64, device=device)
    noise_pred = model(images, t_tensor).sample
    denoised_batch = [
        model.noise_scheduler.step(noise_pred[i], t_tensor[i], images[i]).pred_original_sample
        for i in range(len(images))
    ]
    return torch.stack(denoised_batch)


def plot_batch_results(originals, noisies, denoised_batch, indices, output_dir, path_prev):
    """Plot and save the results for each image in the batch."""
    for i, (original, noisy, denoised, idx) in enumerate(zip(originals, noisies, denoised_batch, indices)):
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))
        denoised = (denoised + 1) / 2.0
        images = [original.transpose(1, 2, 0), noisy.squeeze(0).detach().cpu().numpy(), denoised.squeeze(0).detach().cpu().numpy()]
        titles = ["Original Image", "Noisy Image", "Denoised Image"]
        
        for ax, img, title in zip(axs, images, titles):
            im = ax.imshow(img, cmap="gray")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        
        fig.suptitle(f"Diffusion Sampling (scd_idx={idx})", fontsize=16)
        output_path = os.path.join(output_dir, f"{path_prev}_scd_id={idx}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

def preprocess_labels(images):
    # replace gray pixels into white
    non_black_mask = images > 0
    images[non_black_mask] = 1.0
    return images    

# Main Execution
if __name__ == "__main__":
    # settings
    num_sampling = 5
    timestep = 750
    use_classifier_guidance = True
    # use_black = False
    # use_white = False
    print("use_classifier_guidance: ", use_classifier_guidance)


    # Load models
    diffusion_model = load_model(
        # '/mydata/dlbirhoui/chia/checkpoints/ddim_small_mix/epoch=130-val_loss=0.0160.ckpt',
        # '/mydata/dlbirhoui/chia/checkpoints/ddim_small_more_layer_mix/epoch=219-val_loss=0.0162.ckpt',
        'mydata/dlbirhoui/chia/checkpoints/ddim_small_mean0_mix_one_more_layer/epoch=249-val_loss=0.0175.ckpt',
        DiffusionModel,
        config=TrainingConfig(),
        noise_scheduler=DDIMScheduler(num_train_timesteps=1000, beta_start=1e-5, beta_end=5e-3, beta_schedule='scaled_linear')
    )
    if use_classifier_guidance:
        classifier = load_model('../checkpoints/classifier/unet_classifier/last.ckpt', UnetClassifier)

    # Define batch of images to sample
    data_path = '/mydata/dlbirhoui/firat/OADAT'
    scd_fname_h5 = 'SCD_RawBP.h5'
    scd_key = 'vc_BP'
    batch_indices = np.random.randint(0, 20000, size=num_sampling)  # Adjust batch size as needed
    print(f"Sampling from {scd_fname_h5}, key={scd_key}, indices={batch_indices}")

    # Load batch of images
    scd_image_batch = prepare_images(data_path, scd_fname_h5, scd_key, batch_indices)
    # if scd_key == 'labels':
    #     scd_image_batch = preprocess_labels(scd_image_batch)
    
    # # add noise to black/white image
    # if use_black is True:
    #     batch_indices = "b"
    #     scd_image_batch = torch.ones((1, 1, 256, 256), device=device, dtype=torch.float32)

    # Initialize noise
    num_inference_steps = 1000
    torch.manual_seed(666)
    print('check random seed: ', np.random.randint(1))
    noise = torch.randn(scd_image_batch.shape, device=device)
    noise_scheduler = diffusion_model.noise_scheduler
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
 
    noisy_images = noise_scheduler.add_noise(
        scd_image_batch, noise, torch.full((scd_image_batch.size(0),), timestep, dtype=torch.int64, device=device)
    )

    output_dir = "./"

    if use_classifier_guidance:
        # Sampling with classifier guidance for the batch
        print("Running the classifier-guided sampling loop...")
        classifier_scale = 80.0
        denoised_images = classifier_guided_sampling(diffusion_model, classifier, noisy_images, timestep, classifier_scale=75.0)
        plot_batch_results(scd_image_batch.cpu().numpy(), noisy_images, denoised_images, batch_indices, output_dir, f"mean0_layer_mix_unet_classifier_t={timestep}_s={classifier_scale}")

    else:
        print("Running the sampling loop...")
        denoised_images = sampling(diffusion_model, noisy_images, timestep)    
        plot_batch_results(scd_image_batch.cpu().numpy(), noisy_images, denoised_images, batch_indices, output_dir, f"t={timestep}")
