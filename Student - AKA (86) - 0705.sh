sh download-data.sh 
python data-split.py 
python pretrain.py --pretrainModel vgg19 --epochs 100 --fcDims 2048 2048 --lr 2e-4
python pretrain.py --pretrainModel resnet152 --epochs 100 --fcDims 2048 2048 --lr 2e-4
python pretrain.py --pretrainModel densenet161 --epochs 100 --fcDims 2048 2048 --lr 2e-4
python ensemble.py --fcDims 2048 2048

