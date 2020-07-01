CUDA_VISIBLE_DEVICES=2 python test_ic15_water.py --scale 1 --resume ic15_res50_pretrain_ic17.pth.tar
cd outputs
zip -j submit_ic15.zip submit_ic15/*
cd ../eval
sh eval_ic15.sh