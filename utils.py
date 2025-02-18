# utils.py

import os, glob
from torchvision.transforms import v2
import numpy as np
from diffusers import DDIMScheduler
import torch
import yaml
from types import SimpleNamespace


def get_last_checkpoint(checkpoint_dir: str) -> str:
    latest_ckpt = glob.glob(os.path.join(checkpoint_dir, "last.ckpt"))
    return latest_ckpt[-1] if latest_ckpt else None


def get_named_beta_schedule(schedule_name, num_train_timesteps):
    if schedule_name == "cosine":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )
    elif schedule_name == "cosine_dark":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction"
        )
    elif schedule_name == "cosine_vpred":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction"
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


transforms = v2.Compose(
    [
        v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
        v2.Lambda(lambda x: ((x - (-0.2)) * 2 / 1.2) - 1),
    ]
)

def dict_to_namespace(d):
    """Recursively converts dictionaries to an object with dot notation access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def load_config_from_yaml(yaml_path: str):
    """Load training configuration from a YAML file and convert to dot-accessible object."""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return dict_to_namespace(config_dict["training_config"])  # Convert and return

# # after normalization, the range would be -4.5494004521 to 22.5288367428
# SWFD_MEAN = 0.001611371641047299 # 0.16804751753807068 # 
# SWFD_STD = 0.04431603103876114 # 0.03693051263689995 # 
# SCD_MEAN = 0.0015873634681094503 # 0.1689334511756897 # 
# SCD_STD = 0.04384154212224777 # 0.07449506968259811 # 