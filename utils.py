# utils.py

import os, glob

def get_last_checkpoint(checkpoint_dir: str) -> str:
    latest_ckpt = glob.glob(os.path.join(checkpoint_dir, "last.ckpt"))
    return latest_ckpt[-1] if latest_ckpt else None