import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
import glob
import numpy as np
import torch, torchsummary
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from Modules import utils
from Modules.utils import EventTimer
from Modules.dataset import ProductDataset
from Modules.pretrain import models

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

inferencePreprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def main():
    args = parseArguments()

    testDataset = ProductDataset(os.path.join(args.dataDir, 'test'), os.path.join(args.testImages), transform=inferencePreprocessing)
    testDataloader = DataLoader(testDataset, batch_size=args.batchSize, num_workers=args.numWorkers, shuffle=False)

    models_path = glob.glob(f"{args.modelDir}/*")

    res = open(args.predictFile, 'w')
    res.write("filename,category\n")
    with torch.no_grad():
        for model_path in models_path:
            if model_path.split("/")[-1] == 'resnet50':
                print(model_path)
            else:
                continue
            model = models.GetPretrainedModel(model_path.split("/")[-1], 42, pretrain=True).cuda()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            for i, (data, path) in enumerate(testDataloader):
                test_pred = model(data.cuda())
                test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
                for j in range(len(path)):
                    res.write(f"{path[j]},{test_label[j]}\n")

def parseArguments():
    parser = ArgumentParser()

    parser.add_argument('--numWorkers', type=int, default=8)
    parser.add_argument('--dataDir', default='/tmp3/b06902058/data/')
    parser.add_argument('--modelDir', default='/tmp3/b06902058/models/ensemble')
    parser.add_argument('--predictFile', default='./predict/result.csv')
    parser.add_argument('--testImages', default='/tmp3/b06902058/data/test.pkl')
    parser.add_argument('--batchSize', type=int, default=128)

    return parser.parse_args()

if __name__ == '__main__':
    main()
    
