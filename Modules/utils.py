import os, sys, time
import pickle

import numpy as np

_LightGray = '\x1b[38;5;251m'
_Bold = '\x1b[1m'
_Underline = '\x1b[4m'
_Orange = '\x1b[38;5;215m'
_SkyBlue = '\x1b[38;5;38m'
_Reset = '\x1b[0m'

SEED = 0x06902029 ^ 0x06902049 ^ 0x06902058 ^ 0x06902097

class EventTimer():
    def __init__(self, name = '', verbose = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(_LightGray + '------------------ Begin "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + '" ------------------' + _Reset, file = sys.stderr)
        self.beginTimestamp = time.time()
        return self

    def __exit__(self, type, value, traceback):
        elapsedTime = time.time() - self.beginTimestamp
        if self.verbose:
            print(_LightGray + '------------------ End "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + ' (Elapsed ' + _Orange + f'{elapsedTime:.4f}' + _Reset + 's)" ------------------' + _Reset + '\n', file = sys.stderr)

    def gettime(self):
        return time.time() - self.beginTimestamp

def genPredCSV(filenames, predictions, outputFile, from_prob=False):
    '''
    filenames: A list of N filenames
    predictions: A (N,) shaped integer array if from_prob is False; otherwise, it should be an (N, C) shaped array
    '''
    if from_prob:
        predictions = np.argmax(predictions, axis=1)

    with open(outputFile, 'w') as f:
        print('filename,category', file=f)
        for f, p in zip(filenames, predictions):
            print(f'{f},{p:02d}', file=f)

def pickleSave(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def pickleLoad(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    