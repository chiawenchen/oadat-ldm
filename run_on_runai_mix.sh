# python train.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name ddim_small_mean0_mix --mix_swfd_scd True

python train.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name ddim_small_mean0_mix_cosine --mix_swfd_scd True --noise_schedule cosine