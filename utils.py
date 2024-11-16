# utils.py

import os, glob
from torchvision.transforms import v2
import numpy as np
from diffusers import DDIMScheduler

def get_last_checkpoint(checkpoint_dir: str) -> str:
    latest_ckpt = glob.glob(os.path.join(checkpoint_dir, "last.ckpt"))
    return latest_ckpt[-1] if latest_ckpt else None

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == 'cosine':
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )
    elif schedule_name == 'scaled_linear':
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_timesteps,
            beta_start=1e-5,
            beta_end=5e-3,
            beta_schedule="scaled_linear",
        )
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_timesteps,
            beta_schedule="linear",
        )
    return noise_scheduler

transforms = v2.Compose(
            [
                v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
               
                v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),

                # Scale to range [-1, 1]
                v2.Lambda(lambda x: 2 * x - 1),

            ]
        )

