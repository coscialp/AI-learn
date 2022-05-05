import csv
import numpy as np
from ..linear_model.type import CLASSIFIER, REGRESSOR


class Dataset:
    def __init__(self):
        self.data = None
        self.target = None
        self.features_name = []
        self.targets_name = []


def load_dataset(filename, y_name, type=CLASSIFIER, indesirable_feature=[]):
    dataset = Dataset()
    with open(filename, 'r', newline='') as file:
        csvfile = csv.reader(file)
        first_row = next(csvfile)
        y_index = -1
        index_idx = []
        for index, name in enumerate(first_row):
            if name == y_name:
                y_index = index
            if name in indesirable_feature:
                index_idx.append(index)
            else:
                dataset.features_name.append(name)
        y = []
        X = []
        for row in csvfile:
            if type == CLASSIFIER:
                if row[y_index] not in dataset.targets_name:
                    dataset.targets_name.append(row[y_index])
                row[y_index] = dataset.targets_name.index(row[y_index])
            y.append(row[y_index])
            row.pop(y_index)
            for i in index_idx:
                row.pop(i)
            X.append(row)

    dataset.data = np.array(X, dtype=float)
    dataset.target = np.array(y, dtype=int)
    dataset.target = np.reshape(dataset.target, (dataset.target.shape[0], 1))
    return dataset