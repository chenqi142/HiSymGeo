python train.py --val --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root /data0/chenqi_data/data --data_name CVOGL_SVI --savename test_model_svi --gpu 7,0 --batch_size 8 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root /data0/chenqi_data/data --data_name CVOGL_SVI --savename test_model_svi --gpu 1,7 --batch_size 8 --num_workers 16 --print_freq 50
