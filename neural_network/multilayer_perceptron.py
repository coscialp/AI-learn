"""
Multi-layer Perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class BaseMLP:
    def __init__(self, hidden_layers=(32, 32, 32), n_iter=1000, learning_rate=0.01, normalize=False):
        self.parameters_ = {}
        self.n_iter_ = n_iter
        self.learning_rate_ = learning_rate
        self.loss_ = []
        self.val_loss_ = []
        self.acc_ = []
        self._X = None
        self._y = None
        self.normalize_ = normalize
        self.hidden_layers_ = hidden_layers


    def _initialisation(self, dimensions):
        C = len(dimensions)

        for c in range(1, C):
            self.parameters_[f'W{c}'] = np.random.randn(dimensions[c], dimensions[c - 1])
            self.parameters_[f'b{c}'] = np.random.randn(dimensions[c], 1)



class MLPClassifier(BaseMLP):
    def __init__(self, hidden_layers=(32, 32, 32), n_iter=1000, learning_rate=0.01, normalize=False):
        super().__init__(hidden_layers, n_iter, learning_rate, normalize)

    def _forward_propagation(self, X):
        activations = {'A0': X}

        C = len(self.parameters_) // 2

        for c in range(1, C + 1):
            Z = self.parameters_[f'W{c}'].dot(activations[f'A{c - 1}']) + self.parameters_[f'b{c}']
            activations[f'A{c}'] = 1 / (1 + np.exp(-Z))

        return activations

    def _back_propagation(self, y, activations):
        m = y.shape[1]
        C = len(self.parameters_) // 2
        dZ = activations[f'A{C}'] - y

        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients[f'dW{c}'] = 1 / m * np.dot(dZ, activations[f'A{c - 1}'].T)
            gradients[f'db{c}'] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(self.parameters_[f'W{c}'].T, dZ) * activations[f'A{c - 1}'] * (1 - activations[f'A{c - 1}'])

        return gradients

    def _update(self, gradients):
        C = len(self.parameters_) // 2
        
        for c in range(1, C + 1):
            self.parameters_[f'W{c}'] -= self.learning_rate_ * gradients[f'dW{c}']
            self.parameters_[f'b{c}'] -= self.learning_rate_ * gradients[f'db{c}']

    def log_loss(self, y, A):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def predict_proba(self, X):
        activations = self._forward_propagation(X)
        C = len(self.parameters_) // 2
        Af = activations[f'A{C}']
        return Af

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

    def score(self, X, y):
        return np.sum(np.equal(y, self.predict(X))) / len(y)

    def fit(self, X, y, test_size=0.2, random_state=None):
        dimensions = list(self.hidden_layers_)
        dimensions.insert(0, X.shape[0])
        dimensions.append(y.shape[0])

        self._X = X
        self._y = y
        self._initialisation(dimensions)

        X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=test_size, random_state=random_state)
        X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

        for i in range(self.n_iter_):
            activations = self._forward_propagation(X_train)
            gradients = self._back_propagation(y_train, activations)
            self._update(gradients)

            if i % 10 == 0:
                val_activations = self._forward_propagation(X_test)
                C = len(self.parameters_) // 2
                self.loss_.append(self.log_loss(y_train, activations[f'A{C}']))
                self.val_loss_.append(self.log_loss(y_test, val_activations[f'A{C}']))
                self.acc_.append(self.score(X_test, y_test))

    def display_train(self):
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(self.loss_, label='Loss')
        ax[1].plot(self.acc_, label='Accuracy')
        ax[2].scatter(self._X[0, :], self._X[1, :], c=self._y, s=50)
        x0_lim = ax[2].get_xlim()
        x1_lim = ax[2].get_ylim()

        resolution = 100

        x0 = np.linspace(x0_lim[0], x0_lim[1], resolution)
        x1 = np.linspace(x1_lim[0], x1_lim[1], resolution)

        X0, X1 = np.meshgrid(x0, x1)

        # dim = [X0.ravel(), X1.ravel()]

        dim = [X0.ravel()]

        for i in range(1, self._X.shape[0]):
            dim.append(X1.ravel())

        XX = np.vstack(tuple(dim))

        Z = self.predict(XX)

        Z = Z.reshape((resolution, resolution))

        ax[2].pcolormesh(X0, X1, Z, alpha=0.3, zorder=-1)
        ax[2].contour(X0, X1, Z, colors='green')




    
