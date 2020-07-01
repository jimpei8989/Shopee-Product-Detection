import typing, importlib
from torch import nn
from torchvision import models

# Following https://pytorch.org/docs/stable/torchvision/models.html#

def removeGrads(model):
    for param in model.parameters():
        param.requires_grad = False

def deepFC(lastLayerOutputDim, fcDims):
    network = nn.Sequential()
    for i, dim in enumerate(fcDims):
        network.add_module(f'fc_linear_{i}', nn.Linear(lastLayerOutputDim, dim))
        lastLayerOutputDim = dim

        if i != len(fcDims) - 1:
            network.add_module(f'fc_relu{i}', nn.ReLU())
    return network

def GetPretrainedModel(name: str, numClasses: int, finetune=False, pretrain=True, fcDims=[42]):
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
            model.fc = deepFC(numFeatures, fcDims)
        elif name[:3] == 'vgg':
            # Load model
            model = getattr(models, name)(pretrained=pretrain)

            # Set feature extraction weights require
            if finetune is False:
                removeGrads(model)
            
            # Add classification layer
            numFeatures = model.classifier[-1].in_features
            model.classifier[-1] = deepFC(numFeatures, fcDims)
        elif name[:5] == 'dense':
            # Load model
            model = getattr(models, name)(pretrained=pretrain)

            # Set feature extraction weights require
            if finetune is False:
                removeGrads(model)

            # Add classification layer
            numFeatures = model.classifier.in_features
            model.classifier = deepFC(numFeatures, fcDims)
        else:
            print('! No such pretrained model')
            return None
    except Exception as e:
        print('! Error: Load model failed.')
        print(f'Exception: {e}')
        return None

    return model
