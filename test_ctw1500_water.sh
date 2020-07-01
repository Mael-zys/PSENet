CUDA_VISIBLE_DEVICES=3 python test_ctw1500_water.py --scale 1 --resume checkpoints/ctw1500_resnet50_bs_8_ep_30_pretrain_ic17/checkpoint.pth.tar --long_size 1280
cd eval
sh eval_ctw1500.sh