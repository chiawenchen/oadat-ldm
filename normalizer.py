import os
import torch
from torchvision.transforms import v2
import numpy as np
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

transforms = v2.Compose(
      [
          v2.Lambda(lambda x: np.clip(x / np.max(x), a_min=-0.2, a_max=None)),
      ]
  )

data_path = '/mydata/dlbirhoui/firat/OADAT'
fname = 'SWFD_semicircle_RawBP.h5'
key = 'sc_BP'
indices_swfd = np.load(
    "/mydata/dlbirhoui/chia/oadat-ldm/train_sc_BP_indices.npy"
)

# data_path = '/mydata/dlbirhoui/firat/OADAT'
# fname = 'SCD_RawBP.h5'
# key = 'vc_BP'

# load dataset
obj = dataset.Dataset(
    fname_h5=os.path.join(data_path, fname), key=key, inds=indices_swfd, transforms=transforms
)

dataloader = DataLoader(
    obj,
    batch_size=1000,
    shuffle=False,
    num_workers=4,
)

# Load the entire dataset into memory (use with caution for large datasets)
full_dataset = []
for batch in tqdm(dataloader):
    if isinstance(batch, torch.Tensor):
        batch = batch.numpy()
    full_dataset.append(batch)

full_dataset = np.concatenate(full_dataset, axis=0)  # Combine all batches

# Compute stats for the entire dataset in memory
global_min = np.min(full_dataset)
global_max = np.max(full_dataset)
global_mean = np.mean(full_dataset)
global_std = np.std(full_dataset)

print(f"Global Min: {global_min}, Global Max: {global_max}")
print(f"Global Mean: {global_mean}, Global Std: {global_std}")

# SWFD including test set
# Global Min: -0.20000000298023224, Global Max: 1.0
# Global Mean: 0.0015873753000050783, Global Std: 0.043841637670993805

# SWFD without test set
# Global Min: -0.20000000298023224, Global Max: 1.0
# Global Mean: 0.001611371641047299, Global Std: 0.04431603103876114

# SCD
# Global Min: -0.20000000298023224, Global Max: 1.0
# Global Mean: 0.0029308649245649576, Global Std: 0.0893622562289238