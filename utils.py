# utils.py

import os, glob

def get_last_checkpoint(checkpoint_dir: str) -> str:
    latest_ckpt = glob.glob(os.path.join(checkpoint_dir, "last.ckpt"))
    return latest_ckpt[-1] if latest_ckpt else None


from torchvision.transforms import v2
import numpy as np

transforms = v2.Compose(
            [
                v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
                v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            ]
        )