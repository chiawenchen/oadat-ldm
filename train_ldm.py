import os
import torch
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from datamodule import OADATDataModule
from config.config import LDMTrainingConfig, parse_arguments
from models.LDM import LatentDiffusionModel
from models.VAE import VAE
from utils import get_last_checkpoint, get_named_beta_schedule


# the precision settings for matrix multiplications
torch.set_float32_matmul_precision("medium")


def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()

    # Set up training configuration
    config = LDMTrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    # Set up data module
    # indices_scd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/scd_500px_blob_train_indices.npy")
    indices_swfd = np.load("/mydata/dlbirhoui/chia/oadat-ldm/config/train_sc_BP_indices.npy")
    datamodule = OADATDataModule(
        data_path=args.oadat_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_swfd_scd=args.mix_swfd_scd,
        indices_swfd=indices_swfd,
    )

    # Set up noise scheduler
    noise_scheduler = get_named_beta_schedule(args.noise_schedule, config.num_train_timesteps)

    # Set up sample image saving path
    sample_dir = os.path.join(config.output_dir, "samples", args.job_name)
    os.makedirs(sample_dir, exist_ok=True)
    config.sample_dir = sample_dir

    # Initialize the model
    vae = VAE.load_from_checkpoint(config.vae_ckpt_dir)
    model = LatentDiffusionModel(config=config, noise_scheduler=noise_scheduler, vae=vae)

    # Set up logger
    logger = WandbLogger(
        project="latent-diffusion",
        name=args.job_name,
        log_model=False,
        config=config.__dict__,
    )

    # Set up checkpoint callback
    ckpt_dir = os.path.join(config.output_dir, "checkpoints", args.job_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        save_last=True,
        filename="{epoch:02d}-{val_loss:.4f}",
    )

    # Set up Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)], # depth=-1 for full summary
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )

    # Load the latest checkpoint if available
    latest_ckpt = get_last_checkpoint(ckpt_dir)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
    else:
        print("No checkpoint found, starting from scratch.")

    # Train the model
    trainer.fit(model, datamodule, ckpt_path=latest_ckpt)

if __name__ == "__main__":
    main()
