import torch

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))

def imread(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def img2tensor(img, bgr2rgb=False, float32=True):
	img = img.astype(np.float32) / 255.
	return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
	fig = plt.figure(figsize=(25, 10))
	ax1 = fig.add_subplot(1, 2, 1) 
	plt.title('Input image', fontsize=16)
	ax1.axis('off')
	ax2 = fig.add_subplot(1, 2, 2)
	plt.title('NAFNet output', fontsize=16)
	ax2.axis('off')
	ax1.imshow(img1)
	ax2.imshow(img2)

def single_image_inference(model, img, save_path):
	model.feed_data(data={'lq': img.unsqueeze(dim=0)})

	if model.opt['val'].get('grids', False):
			model.grids()

	model.test()

	if model.opt['val'].get('grids', False):
			model.grids_inverse()

	visuals = model.get_current_visuals()
	sr_img = tensor2img([visuals['result']])
	imwrite(sr_img, save_path)

def init_model():	
	opt_path = os.path.join(DIR, os.path.normpath('NAFNet-width64.yml'))
	opt = parse(opt_path, is_train=False)
	opt['dist'] = False
	NAFNet = create_model(opt)
	return NAFNet

def deblur(input_path, output_path, nafnet):
	img_input = imread(input_path)
	inp = img2tensor(img_input)
	single_image_inference(nafnet, inp, output_path)