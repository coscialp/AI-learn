import numpy as np

def StandartScaler(X, mean=None, std=None):
    try:
        if mean == None:
            mean = X.mean(axis=1)
        if std == None:
            std = X.std(axis=1)
    except ValueError:
        pass
    
    np_mean = np.tile(mean, (X.shape[1], 1)).T
    np_std = np.tile(std, (X.shape[1], 1)).T
    return (X - np_mean) / np_std

def MinMaxScaler(X, min=None, max=None):
    try:
        if min == None:
            min = X.min(axis=1)
        if max == None:
            max = X.max(axis=1)
    except ValueError:
        pass

    np_min = np.tile(min, (X.shape[1], 1)).T
    np_max = np.tile(max, (X.shape[1], 1)).T
    return (X - np_min) / (np_max - np_min)
