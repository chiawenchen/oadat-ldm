import os
from config import TrainingConfig
import matplotlib.pyplot as plt
import torch
from diffusers import DDIMScheduler
import dataset
from torch.utils.data import DataLoader
from utils import transforms, get_last_checkpoint
import math
import numpy as np
from numpy import random
from model import DiffusionModel
import torch.nn.functional as F
from train_classifier import ResNetClassifier
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

# Load the diffusion model checkpoint
diffusion_ckpt_path = '../checkpoints/ddim_small_clip_1e-5_5e-3/epoch=189-val_loss=0.0216.ckpt'
diffusion_model = DiffusionModel.load_from_checkpoint(
    checkpoint_path=diffusion_ckpt_path, config=TrainingConfig(), noise_scheduler=DDIMScheduler(
        num_train_timesteps=1000, beta_start=1e-5, beta_end=5e-3, beta_schedule='scaled_linear'
    )
)
diffusion_model.to(device)
diffusion_model.eval()

# Load the classifier checkpoint
classifier_ckpt_path = '../checkpoints/classifier/resnet_classifier/epoch=68-val_loss=0.0075.ckpt'
classifier = ResNetClassifier.load_from_checkpoint(classifier_ckpt_path)
classifier.to(device)
classifier.eval()

# Extract a sample image
data_path = '/mydata/dlbirhoui/firat/OADAT'
scd_fname_h5 = 'SCD_RawBP.h5'
scd_key = 'vc_BP'
random_idx = random.randint(0, 20000)
print(f"Sampling from {scd_fname_h5}, key={scd_key}, idx={random_idx}")

# Load sample image from dataset
scd_obj = dataset.Dataset(
    fname_h5=os.path.join(data_path, scd_fname_h5), key=scd_key, transforms=transforms, inds=[random_idx]
)
scd_image = next(iter(scd_obj))
scd_image_tensor = torch.tensor(scd_image, device=device, dtype=torch.float32).unsqueeze(0)

# Define noise scheduler
noise_scheduler = diffusion_model.noise_scheduler
noise_scheduler.set_timesteps(num_inference_steps=100)

# Prepare noisy starting image
timestep = torch.tensor([100], dtype=torch.int64, device=device)
noise = torch.randn(scd_image_tensor.shape, device=device)
noisy_image = noise_scheduler.add_noise(scd_image_tensor, noise, timestep)
denoised_image = noisy_image

# Classifier-guided sampling loop
print("Running the classifier-guided sampling loop...")
for t in tqdm(reversed(noise_scheduler.timesteps), desc="Sampling with Classifier Guidance"):
    # Ensure timestep is on the correct device and integer type
    t = torch.tensor([t], dtype=torch.int64, device=device)
    
    # Predict noise using the diffusion model
    noise_pred = diffusion_model(denoised_image, t).sample
    
    # Calculate classifier gradient
    denoised_image.requires_grad_(True)
    logit = classifier(denoised_image, t)
    target_label = torch.tensor([1], dtype=torch.long, device=device)  # Target class label
    loss = F.cross_entropy(logit, target_label)
    classifier_grad = torch.autograd.grad(loss, denoised_image)[0]
    
    # Adjust noise prediction based on classifier gradient
    guidance_scale = 1.0  # Scale to control classifier influence
    noise_pred = noise_pred - guidance_scale * classifier_grad
    
    # Perform denoising step
    step_output = noise_scheduler.step(noise_pred, t, denoised_image)
    denoised_image = step_output.prev_sample  # Update noisy image for next timestep

print('plotting the result...')
# Show original, noisy, and denoised images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axs[0].imshow(scd_image.transpose(1, 2, 0), cmap='gray')
axs[0].set_title(f"{scd_key} {scd_fname_h5} Image at idx={random_idx}")

# Noisy image at specified timestep
axs[1].imshow(noisy_image[0].squeeze(0).detach().cpu().numpy(), cmap='gray')
axs[1].set_title(f"Noisy Image at timestep {timestep.item()}")

# Denoised image after classifier-guided sampling
axs[2].imshow(denoised_image[0].squeeze(0).detach().cpu().numpy(), cmap='gray')
axs[2].set_title("Classifier-Guided Denoised Image")

output_path = f"classifier_guided_diffusion_result_idx={random_idx}.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close()  
