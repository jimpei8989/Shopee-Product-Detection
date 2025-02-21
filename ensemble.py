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
from Modules.utils import EventTimer, genPredCSV
from Modules.dataset import ProductDataset
from Modules.pretrain import models

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

inferencePreprocessing = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop((224, 224), scale = (0.75, 1), ratio = (0.8, 1.25)),
    transforms.ColorJitter(brightness = 0.15, contrast = 0.15, saturation = 0.15, hue = 0.1),
    transforms.ToTensor(),
    normalize
])

def main():
    args = parseArguments()

    testDataset = ProductDataset(os.path.join(args.dataDir, 'test'), os.path.join(args.testImages), transform=inferencePreprocessing)
    testDataloader = DataLoader(testDataset, batch_size=args.batchSize, num_workers=args.numWorkers, shuffle=False)

    models_path = glob.glob(f"{args.modelDir}/*")

    filenames = utils.pickleLoad(args.testImages)
    predictions = np.zeros((len(filenames), 42))

    with torch.no_grad():
        pred = []
        for model_path in models_path:
            model = models.GetPretrainedModel(model_path.split("/")[-1].split('-')[0], fcDims=args.fcDims+[42]).cuda()
            model.load_state_dict(torch.load(model_path))
            model.eval()

            with EventTimer(f'Predicting {model_path}'):
                for epoch in range(args.ttaEpoch):
                    predList = []
                    for data, path in tqdm(testDataloader):
                        predList.append(model(data.cuda()).cpu().data.numpy())

                    predictions += np.concatenate(predList, axis=0)
            
            del model

    genPredCSV(filenames, predictions, args.predictFile, from_prob=True)

def parseArguments():
    parser = ArgumentParser()

    parser.add_argument('--numWorkers', type=int, default=8)
    parser.add_argument('--dataDir', default='data/')
    parser.add_argument('--modelDir', default='ensemble')
    parser.add_argument('--predictFile', default='result.csv')
    parser.add_argument('--testImages', default='data/test.pkl')
    parser.add_argument('--batchSize', type=int, default=256)
    parser.add_argument('--ttaEpoch', type=int, default=10)
    parser.add_argument('--fcDims', type=int, nargs='+', default=[], help='Do not include output dimension')

    return parser.parse_args()

if __name__ == '__main__':
    main()
    
