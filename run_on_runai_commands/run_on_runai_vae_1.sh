# python train_vae_aekl.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 128 --num_workers 4 --job_name aekl --mix_swfd_scd
# wandb agent chiachen-eth-z-rich/vae/picqh8iz
# python train_vae_aekl.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 128 --num_workers 4 --job_name aekl_2atten --mix_swfd_scd
python train_vae_aekl.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 128 --num_workers 4 --job_name aekl_latent_size128 --mix_swfd_scd
