import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from Modules import utils

class ProductDataset(Dataset):
    def __init__(self, rootDir, dataPath, transform=None, data=None):
        super().__init__()
        self.rootDir = rootDir
        if data is not None:
            pass
        else:
            data = utils.pickleLoad(dataPath)
        if len(data) == 2:
            self.imgPathes = data[0]
            self.labels = torch.LongTensor(data[1])
        else:
            self.imgPathes = data
            self.labels = None

        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.rootDir, self.imgPathes[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img, self.imgPathes[idx]
        else:
            return img, self.labels[idx], self.imgPathes[idx]

    def __len__(self):
        return len(self.imgPathes)

    def remove(idx):
        self.imgPathes.pop(idx)
        self.labels.pop(idx) 

