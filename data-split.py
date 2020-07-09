import os
from collections import Counter
from sklearn.model_selection import train_test_split

from Modules.utils import EventTimer
from Modules import utils

'''
Simply analysis the dataset and splits training data and validation data

The training / validation data will be in the following format:
    - X: A list of relative pathes (from 'data/train/'), representing the image
    - Y: A list of integers, representing the corresponding category

The testing data will be in the following format:
    - X: A list of relative pathes (from 'data/test/'), representing the image
'''

def main():
    with EventTimer('Handling Training Data'):
        with open('data/train.csv') as f:
            data = list(map(lambda p : (os.path.join(p[1], p[0]), int(p[1])), map(lambda s : s.strip().split(','), f.readlines()[1:])))

            counter = Counter(map(lambda d : d[1], data))

            print(f'> Dataset Size: {len(data)}')
            print(f'> Categories Counts')
            print(f'> {counter}')

        trainingRatio = 0.8

        X, Y = zip(*data)
        trainX, validX, trainY, validY = train_test_split(X, Y, train_size=trainingRatio, random_state=utils.SEED)

        print(f'> Training Size:    {len(trainX)}')
        print(f'> Validation Size:  {len(validX)}')

        utils.pickleSave((trainX, trainY), 'data/train.pkl')
        utils.pickleSave((validX, validY), 'data/valid.pkl')
        utils.pickleSave((trainX + validX, trainY + validY), 'data/train-all.pkl')
        utils.pickleSave(([], []), 'data/valid-all.pkl')

    with EventTimer('Handling Testing Data'):
        with open('data/test.csv') as f:
            data = list(map(lambda s : s.strip().split(',')[0], f.readlines()[1:]))

        print(f'> Testing Size: {len(data)}')
        utils.pickleSave(data, 'data/test.pkl')

if __name__ == '__main__':
    main()
