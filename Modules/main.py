from utils import *
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		# torch.nn.MaxPool2d(kernel_size, stride, padding)
		# input 維度 [3, 128, 128]
		self.cnn = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, 1),  # [32, 128, 128]
			nn.Conv2d(32, 32, 3, 1, 1),  # [32, 128, 128]
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	  # [32, 64, 64]

			nn.Conv2d(32, 64, 3, 1, 1), # [64, 64, 64]
			nn.Conv2d(64, 64, 3, 1, 1), # [64, 64, 64]
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	  # [64, 32, 32]

			nn.Conv2d(64, 128, 3, 1, 1), # [128, 32, 32]
			nn.Conv2d(128, 128, 3, 1, 1), # [128, 32, 32]
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	  # [128, 16, 16]

			nn.Conv2d(128, 256, 3, 1, 1), # [256, 16, 16]
			nn.Conv2d(256, 256, 3, 1, 1), # [256, 16, 16]
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	   # [512, 8, 8]
			
			nn.Conv2d(256, 512, 3, 1, 1), # [512, 8, 8]
			nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	   # [512, 4, 4]
		)
		self.fc = nn.Sequential(
			nn.Linear(512*4*4, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 42)
		)

	def forward(self, x):
		out = self.cnn(x)
		out = out.view(out.size()[0], -1)
		return self.fc(out)

class ImgDataset(Dataset):
	def __init__(self, x, y=None, transform=None):
		self.x = x
		# label is required to be a LongTensor
		self.y = y
		if y is not None:
			self.y = torch.LongTensor(y)
		self.transform = transform
	def __len__(self):
		return len(self.x)
	def __getitem__(self, index):
		X = self.x[index]
		if self.transform is not None:
			X = self.transform(X)
		if self.y is not None:
			Y = self.y[index]
			return X, Y
		else:
			return X

def train():
	train_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
		transforms.RandomRotation(15), # 隨機旋轉圖片
		transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
	])
	# testing 時不需做 data augmentation
	test_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
	])

	with EventTimer('Loading Data'):
		train_X = np.load(f"{data_dir}/train_X.npy")
		train_Y = np.load(f"{data_dir}/train_Y.npy")
		valid_X = np.load(f"{data_dir}/valid_X.npy")
		valid_Y = np.load(f"{data_dir}/valid_Y.npy")


	train_set = ImgDataset(train_X, train_Y, train_transform)
	valid_set = ImgDataset(valid_X, valid_Y, test_transform)
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)


	with EventTimer('Start Training'):
		model = Classifier().cuda()
		loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
		num_epoch = 30

		for epoch in range(num_epoch):
			epoch_start_time = time.time()
			train_acc = 0.0
			train_loss = 0.0
			valid_acc = 0.0
			valid_loss = 0.0

			model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
			for i, data in enumerate(train_loader):
				optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
				train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
				batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
				batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
				optimizer.step() # 以 optimizer 用 gradient 更新參數值

				train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
				train_loss += batch_loss.item()

			model.eval()
			with torch.no_grad():
				for i, data in enumerate(valid_loader):
					valid_pred = model(data[0].cuda())
					batch_loss = loss(valid_pred, data[1].cuda())

					valid_acc += np.sum(np.argmax(valid_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
					valid_loss += batch_loss.item()

				#將結果 print 出來
				print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
					(epoch + 1, num_epoch, time.time()-epoch_start_time, \
					 train_acc/train_set.__len__(), train_loss/train_set.__len__(), valid_acc/valid_set.__len__(), valid_loss/valid_set.__len__()))


if __name__ =='__main__':
	data_dir = "/tmp3/b06902058/data/"
	batch_size = 128
	train()
