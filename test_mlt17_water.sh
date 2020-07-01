CUDA_VISIBLE_DEVICES=2 python test_mlt17_water.py --scale 1 --resume checkpoints/ic15_resnet50_bs_8_ep_30_pretrain_ic17/checkpoint.pth.tar
cd outputs_mlt
zip -j submit_mlt.zip submit_ic15/*