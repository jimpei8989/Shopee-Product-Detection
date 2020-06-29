import os
import numpy as np
from utils import *
import cv2

"""
Do preprocessing

Will generate data/train_X.npy, data/train_Y.npy, data/valid_X.npy, data_valid_Y.npy

TODO: config.ini : image_size, data_path ...etc

"""
def pkl2npy():
	train_pkl = pickleLoad(f"{data_dir}/train.pkl")
	valid_pkl = pickleLoad(f"{data_dir}/valid.pkl")
	train_X = np.zeros((len(train_pkl[0]), 128, 128, 3), dtype=np.uint8)
	train_Y = np.zeros((len(train_pkl[0])), dtype=np.uint8)
	valid_X = np.zeros((len(valid_pkl[0]), 128, 128, 3), dtype=np.uint8)
	valid_Y = np.zeros((len(valid_pkl[0])), dtype=np.uint8)

	for i in range(len(train_pkl[0])):
		path = f"{data_dir}/train/{train_pkl[0][i]}"
		label = int(train_pkl[1][i])
		img = cv2.imread(path)
		train_X[i, :, :] = cv2.resize(img,(128, 128))
		train_Y[i] = label

	for i in range(len(valid_pkl[0])):
		path = f"{data_dir}/train/{valid_pkl[0][i]}"
		label = int(valid_pkl[1][i])
		img = cv2.imread(path)
		valid_X[i, :, :] = cv2.resize(img,(128, 128))
		valid_Y[i] = label

	np.save(f"{data_dir}/train_X", train_X)
	np.save(f"{data_dir}/train_Y", train_Y)
	np.save(f"{data_dir}/valid_X", valid_X)
	np.save(f"{data_dir}/valid_Y", valid_Y)
	

if __name__ == '__main__':
	data_dir = "/tmp3/b06902058/data/"
	pkl2npy()
