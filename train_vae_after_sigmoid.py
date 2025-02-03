import os
import numpy as np
import wandb
from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from config.parser import parse_arguments
from utils import get_last_checkpoint, get_named_beta_schedule, load_config_from_yaml
import dataset
from datamodule import OADATDataModule

from models.VAE_after_sigmoid import VAE
from models.CVAE_after_sigmoid import CVAE

# Set precision for matrix multiplications
torch.set_float32_matmul_precision("medium")


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load training configuration from YAML as a dot-accessible object
    config = load_config_from_yaml(args.config_path)

    # Load dataset indices
    if not os.path.exists(config.dataset.scd_train_indices):
        raise FileNotFoundError(f"Dataset file not found: {config.dataset.scd_train_indices}")
    if not os.path.exists(config.dataset.swfd_train_indices):
        raise FileNotFoundError(f"Dataset file not found: {config.dataset.swfd_train_indices}")

    scd_train_indices = np.load(config.dataset.scd_train_indices)
    swfd_train_indices = np.load(config.dataset.swfd_train_indices)

    # Initialize data module
    datamodule = OADATDataModule(
        data_path=args.oadat_dir,  # Using CLI argument
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mix_swfd_scd=config.dataset.mix_swfd_scd,
        indices_scd=scd_train_indices,
        indices_swfd=swfd_train_indices,
        return_labels=config.use_domain_classifier
    )

    # Set up sample and fixed noisy image directories
    os.makedirs(config.paths.sample_dir, exist_ok=True)

    # Initialize the VAE
    if config.cvae:
        model = CVAE(config=config)
    else:
        model = VAE(config=config)

    # Set up WandB logger
    logger = WandbLogger(
        project=config.wandb.project_name,
        name=config.wandb.job_name,
        log_model=False,
        config=vars(config),
    )

    # Set up checkpoint callback
    os.makedirs(config.paths.vae_ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.vae_ckpt_dir,
        save_top_k=config.checkpointing.save_top_k,
        monitor=config.checkpointing.monitor_metric,
        mode=config.checkpointing.monitor_mode,
        save_weights_only=False,
        save_last=config.checkpointing.save_last,
        filename=config.checkpointing.filename_format,
    )

    # Set up Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=1,
    )

    # Load the latest checkpoint if available
    latest_ckpt = get_last_checkpoint(config.paths.vae_ckpt_dir)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        trainer.fit(model, datamodule, ckpt_path=latest_ckpt)
    else:
        print("No checkpoint found, starting from scratch.")
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
