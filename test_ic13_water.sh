CUDA_VISIBLE_DEVICES=2 python test_ic13_water.py --long_size 1600 --scale 1 --resume checkpoints/ic15_resnet50_bs_8_ep_60_pretrain_ic17/checkpoint.pth.tar
python eval_ic13.py