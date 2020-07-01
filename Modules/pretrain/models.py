import typing, importlib
from torch import nn
from torchvision import models

# Following https://pytorch.org/docs/stable/torchvision/models.html#

def removeGrads(model):
    for param in model.parameters():
        param.requires_grad = False

def GetPretrainedModel(name: str, numClasses: int, finetune=False, pretrain=True):
    '''
    Generates a pretrained image classification model
    '''
    try:
        if name[:6] == 'resnet':
            # Load model
            model = getattr(models, name)(pretrained=pretrain)

            # Set feature extraction weights require
            if finetune is False:
                removeGrads(model)
            
            # Add classification layer
            numFeatures = model.fc.in_features
            model.fc = nn.Linear(numFeatures, 1000)
            model = nn.Sequential( model,
                                   nn.Linear(1000, numClasses)
                    )
        elif name[:3] == 'vgg':
            # Load model
            model = getattr(models, name)(pretrained=pretrain)

            # Set feature extraction weights require
            if finetune is False:
                removeGrads(model)
            
            # Add classification layer
            numFeatures = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(numFeatures, numClasses)
        elif name[:5] == 'dense':
            # Load model
            model = getattr(models, name)(pretrained=pretrain)

            # Set feature extraction weights require
            if finetune is False:
                removeGrads(model)

            # Add classification layer
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, numClasses)
        else:
            print('! No such pretrained model')
            return None
    except:
        print('! Error: Load model failed.')
        return None

    return model
