import csv
import numpy as np
from ..linear_model.type import CLASSIFIER, REGRESSOR


class Dataset:
    def __init__(self):
        self.data = None
        self.target = None
        self.features_name = []
        self.targets_name = []

    def describe(self, indesirable_feature=[]):
        describe = {
            'title': "\t|",
            'count': "Count\t|",
            'mean': "Mean\t|",
            'std': "Std\t|",
            'min': "Min\t|",
            '25%': "25%\t|",
            '50%': "50%\t|",
            '75%': "75%\t|",
            'max': "Max\t|",
            }

        count = self.data.shape[0]
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        min = self.data.min(axis=0)
        max = self.data.max(axis=0)
        q1 = np.percentile(self.data, 25, axis=0)
        med = np.percentile(self.data, 50, axis=0)
        q3 = np.percentile(self.data, 75, axis=0)

        idx = 0
        for feature_name in self.features_name:
            if feature_name in indesirable_feature:
                continue
            describe['title'] += '%*s |' % (len(feature_name)+ 3, feature_name)
            describe['count'] += '%*d |' % (len(feature_name)+ 3, count)
            describe['mean'] += '%*.3f |' % (len(feature_name)+ 3, mean[idx])
            describe['std'] += '%*.3f |' % (len(feature_name)+ 3, std[idx])
            describe['min'] += '%*.3f |' % (len(feature_name)+ 3, min[idx])
            describe['max'] += '%*.3f |' % (len(feature_name)+ 3, max[idx])
            describe['25%'] += '%*.3f |' % (len(feature_name)+ 3, q1[idx])
            describe['75%'] += '%*.3f |' % (len(feature_name)+ 3, med[idx])
            describe['50%'] += '%*.3f |' % (len(feature_name)+ 3, q3[idx])
            idx += 1

        for k, v in describe.items():
            print(v)


def load_dataset(filename, y_name, type=CLASSIFIER, indesirable_feature=[], features_name=None):
    dataset = Dataset()
    with open(filename, 'r', newline='') as file:
        csvfile = csv.reader(file)
        if features_name == None:
            first_row = next(csvfile)
        else:
            first_row = features_name
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