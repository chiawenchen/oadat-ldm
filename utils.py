# utils.py

import os, glob
from torchvision.transforms import v2
import numpy as np
from diffusers import DDIMScheduler
import torch


def get_last_checkpoint(checkpoint_dir: str) -> str:
    latest_ckpt = glob.glob(os.path.join(checkpoint_dir, "last.ckpt"))
    return latest_ckpt[-1] if latest_ckpt else None


def get_named_beta_schedule(schedule_name, num_train_timesteps):
    if schedule_name == "cosine":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )
    elif schedule_name == "cosine_noclip":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False
        )
    elif schedule_name == "cosine_dark":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction"
        )
    elif schedule_name == "cosine_rescale":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            rescale_betas_zero_snr=True,
        )
    elif schedule_name == "cosine_vpred":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction"
        )
    elif schedule_name == "cosine_rescale_trailing":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
    elif schedule_name == "scaled_linear":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-5,
            beta_end=5e-3,
            beta_schedule="scaled_linear",
        )
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
        )
    return noise_scheduler

# after normalization, the range would be -4.5494004521 to 22.5288367428
SWFD_MEAN = 0.001611371641047299 # 0.16804751753807068 # 
SWFD_STD = 0.04431603103876114 # 0.03693051263689995 # 
SCD_MEAN = 0.0015873634681094503 # 0.1689334511756897 # 
SCD_STD = 0.04384154212224777 # 0.07449506968259811 # 

swfd_norm_transforms = v2.Compose(
    [
        v2.Lambda(lambda x: np.squeeze(x, axis=0) if x.shape[0] == 1 else x),  # Remove unnecessary channel for grayscale
        v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),  # Normalize and clip
        v2.Lambda(lambda x: np.expand_dims(x, axis=-1) if len(x.shape) == 2 else x),  # Ensure HWC format for ToTensor
        v2.ToImage(),  # Convert to tensor (HWC → CHW)
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([SWFD_MEAN], [SWFD_STD]),  # Normalize using mean and std
    ]
)

scd_norm_transforms = v2.Compose(
    [
        v2.Lambda(lambda x: np.squeeze(x, axis=0) if x.shape[0] == 1 else x),  # Remove unnecessary channel for grayscale
        v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),  # Normalize and clip
        v2.Lambda(lambda x: np.expand_dims(x, axis=-1) if len(x.shape) == 2 else x),  # Ensure HWC format for ToTensor
        v2.ToImage(),  # Convert to tensor (HWC → CHW)
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([SCD_MEAN], [SCD_STD]),  # Normalize using mean and std
    ]
)

swfd_transforms = v2.Compose(
    [
        v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
        v2.Lambda(lambda x: ((x - (-0.2)) * 2 / 1.2) - 1),
    ]
)

scd_transforms = v2.Compose(
    [
        v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
        v2.Lambda(lambda x: ((x - (-0.2)) * 2 / 1.2) - 1),
    ]
)

transforms = v2.Compose(
    [
        v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
        v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        # Scale to range [-1, 1]
        v2.Lambda(lambda x: 2 * x - 1),
    ]
)


# channel3_swfd_transforms = v2.Compose(
#     [
#         v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
#         v2.Lambda(lambda x: torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1)),
#         v2.Normalize([SWFD_MEAN] * 3, [SWFD_STD] * 3),
#     ]
# )

# channel3_scd_transforms = v2.Compose(
#     [
#         v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
#         v2.Lambda(lambda x: torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1)),
#         v2.Normalize([SCD_MEAN] * 3, [SCD_STD] * 3),
#     ]
# )
