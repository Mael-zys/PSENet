import time
import torch
import subprocess
import os
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../../')
	os.chdir('../../')
	res = subprocess.getoutput('python ./det_ic13/script.py –g=./det_ic13/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	# print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	model_name = './pths/east_vgg16.pth'
	test_img_path = os.path.abspath('/home/zhangyangsong/OCR/ICDAR2015/test_im')
	submit_path = 'outputs/submit_ic15'
	eval_model(model_name, test_img_path, submit_path)
