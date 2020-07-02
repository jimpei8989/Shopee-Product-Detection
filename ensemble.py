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
    first_model = 0

    with torch.no_grad():
        file_name = []
        pred = []
        for model_path in models_path:
            model = models.GetPretrainedModel(model_path.split("/")[-1], fcDims=args.fcDims+[42]).cuda()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            cnt = 0
            for i, (data, path) in enumerate(testDataloader):
                test_pred = model(data.cuda())
                test_prob = test_pred.cpu().data.numpy()
                if not first_model:
                    for j in range(len(path)):
                        file_name.append(path[j])
                        pred.append(test_prob[j])
                else:
                    for j in range(len(path)):
                        for k in range(len(test_prob[j])):
                            pred[cnt][k] += test_prob[j][k]
                        cnt += 1

            first_model = 1
        genPredCSV(file_name, pred, args.predictFile, from_prob=True)

def parseArguments():
    parser = ArgumentParser()

    parser.add_argument('--numWorkers', type=int, default=8)
    parser.add_argument('--dataDir', default='data/')
    parser.add_argument('--modelDir', default='ensemble')
    parser.add_argument('--predictFile', default='predict/result.csv')
    parser.add_argument('--testImages', default='data/test.pkl')
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--fcDims', type=int, nargs='+', default=[], help='Do not include output dimension')

    return parser.parse_args()

if __name__ == '__main__':
    main()
    
