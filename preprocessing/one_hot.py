import numpy as np


def one_hot(y):
    y_one_hot = np.zeros((y.shape[0], np.max(y) + 1), dtype=int)
    for i, val in enumerate(y):
        y_one_hot[i, val] = 1
    return y_one_hot