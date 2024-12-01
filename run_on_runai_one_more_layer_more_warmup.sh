# python train_one_more_layer.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm --noise_schedule cosine
python train_one_more_layer_more_warmup.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm_shift_10warmup --noise_schedule cosine

