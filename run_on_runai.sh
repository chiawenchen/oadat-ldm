# python train.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name ddim_small_mean0 --mix_swfd_scd False
python train.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name ddim_small_mean0 --mix_swfd_scd False --noise_schedule cosine
