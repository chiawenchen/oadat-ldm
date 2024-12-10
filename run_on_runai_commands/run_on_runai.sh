# python train_one_more_layer.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm --noise_schedule cosine
# python train_one_more_layer.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm_shift --noise_schedule cosine
# python train_one_more_layer.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm_shift_dark --noise_schedule cosine_dark
# python train_one_more_layer.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm_shift_noclip --noise_schedule cosine_noclip
# python train_one_more_layer_v.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm_shift_mix_v --mix_swfd_scd --noise_schedule cosine_vpred
python train_one_more_layer_v.py --oadat_dir /mydata/dlbirhoui/firat/OADAT --num_epochs 250 --gpus 1 --batch_size 32 --num_workers 4 --job_name dm_shift_v_rescale --noise_schedule cosine_dark
