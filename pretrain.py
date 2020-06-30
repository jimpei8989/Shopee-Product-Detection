import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

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

def accuracy(yPred, yTruth):
    return np.mean(np.argmax(yPred, axis = 1) == yTruth)

# Following https://github.com/pytorch/vision/issues/39
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trainingPreprocessing = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop((224, 224), scale = (0.75, 1), ratio = (0.8, 1.25)),
    transforms.ColorJitter(brightness = 0.15, contrast = 0.15, saturation = 0.15, hue = 0.1),
    transforms.ToTensor(),
    normalize
])

inferencePreprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def main():
    args = parseArguments()

    os.makedirs(args.modelDir, exist_ok=True)
    checkpointDir = os.path.join(args.modelDir, 'checkpoints')
    os.makedirs(checkpointDir, exist_ok=True)

    with EventTimer('Preparing for dataset / dataloader'):
        trainDataset = ProductDataset(os.path.join(args.dataDir, 'train'), os.path.join(args.trainImages), transform=trainingPreprocessing)
        validDataset = ProductDataset(os.path.join(args.dataDir, 'train'), os.path.join(args.validImages), transform=inferencePreprocessing)

        trainDataloader = DataLoader(trainDataset, batch_size=args.batchSize, num_workers=args.numWorkers, shuffle=True)
        validDataloader = DataLoader(validDataset, batch_size=args.batchSize, num_workers=args.numWorkers, shuffle=False)

        print(f'> Training dataset:\t{len(trainDataset)}')
        print(f'> Validation dataset:\t{len(validDataset)}')

    with EventTimer(f'Load pretrained model - {args.pretrainModel}'):
        model = models.GetPretrainedModel(args.pretrainModel, 42)
        #torchsummary will crash under densenet, skip the summary.
        #torchsummary.summary(model, (3, 224, 224), device='cpu')

    with EventTimer(f'Train model'):
        model.cuda()

        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, threshold=1e-2, patience=10)
        history = []

        if args.retrain != 0:
            checkpoint = torch.load(os.path.join(checkpointDir, f'checkpoint-{args.retrain:03d}.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            history = checkpoint['history']

        def runEpoch(dataloader, train=False, name=''):
            # Enable grad
            with (torch.enable_grad() if train else torch.no_grad()):
                if train: model.train()
                else: model.eval()

                losses = []
                for x, y in tqdm(dataloader, desc=name, dynamic_ncols=True):
                    if train:
                        optimizer.zero_grad()
                    
                    output = model(x.cuda()).cpu()
                    loss = criterion(output, y)

                    if train:
                        loss.backward()
                        optimizer.step()

                    accu = accuracy(output.data.numpy(), y.numpy())
                    losses.append((loss.item(), accu))

            return map(np.mean, zip(*losses))

        for epoch in range(args.retrain + 1, args.epochs + 1):
            with EventTimer(verbose=False) as et:
                print(f'====== Epoch {epoch:3d} / {args.epochs:3d} ======')
                trainLoss, trainAccu = runEpoch(trainDataloader, train = True, name='training  ')
                validLoss, validAccu = runEpoch(validDataloader, name='validation')

                history.append(((trainLoss, trainAccu), (validLoss, validAccu)))

                scheduler.step(validLoss)
                print(f'[{et.gettime():.4f}s] Training: {trainLoss:.6f} / {trainAccu:.4f} ; Validation {validLoss:.6f} / {validAccu:.4f}')

            if epoch % 5 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                }, os.path.join(checkpointDir, f'checkpoint-{epoch:03d}.pt'))

        # save model as its coressponding name
        torch.save(model.state_dict(), os.path.join(args.modelDir, args.pretrainModel))
        utils.pickleSave(history, os.path.join(args.modelDir, 'history.pkl'))

def parseArguments():
    parser = ArgumentParser()

    parser.add_argument('--numWorkers', type=int, default=8)
    parser.add_argument('--dataDir', default='/tmp3/b06902058/data/')
    parser.add_argument('--modelDir', default=f'/tmp3/b06902058/models/{datetime.now().strftime("%m%d-%H%M")}')
    parser.add_argument('--trainImages', default='/tmp3/b06902058/data/train.pkl')
    parser.add_argument('--validImages', default='/tmp3/b06902058/data/valid.pkl')
    parser.add_argument('--batchSize', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--retrain', type=int, default=0)

    parser.add_argument('--pretrainModel', default='resnet152')
    return parser.parse_args()

if __name__ == '__main__':
    main()
    
