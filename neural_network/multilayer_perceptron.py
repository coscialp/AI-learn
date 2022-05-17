"""
Multi-layer Perceptron
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ..preprocessing import shuffle_in_unison, StandartScaler


class BaseMLP:
    def __init__(
        self,
        hidden_layers=(32, 32, 32),
        n_iter=1000,
        learning_rate=0.01,
        normalize=False,
        momentum=0.9,
        nesterov=True,
        early_stopping=False,
        multiclass=False
        ):
        self.parameters_ = {}
        self.optimizers_ = {}
        self.n_iter_ = n_iter
        self.learning_rate_ = learning_rate
        self.loss_ = []
        self.val_loss_ = []
        self.acc_ = []
        self._X = None
        self._y = None
        self.normalize_ = normalize
        self.normalize_mean_ = 0
        self.normalize_std_ = 0
        self.hidden_layers_ = hidden_layers
        self.momentum_ = momentum
        self.nesterov_ = nesterov
        self.early_stopping_ = early_stopping
        self.multiclass_ = multiclass


    def _initialisation(self, dimensions):
        C = len(dimensions)

        for c in range(1, C):
            self.parameters_[f'W{c}'] = np.random.randn(dimensions[c], dimensions[c - 1])
            self.parameters_[f'b{c}'] = np.random.randn(dimensions[c], 1)
            self.optimizers_[f'mdW{c}'] = 0
            self.optimizers_[f'mdb{c}'] = 0
            self.optimizers_[f'vdW{c}'] = 0
            self.optimizers_[f'vdb{c}'] = 0


class MLPClassifier(BaseMLP):
    def __init__(
        self,
        hidden_layers=(32, 32, 32),
        n_iter=1000,
        learning_rate=0.01,
        normalize=False,
        momentum=0.9,
        nesterov=True,
        early_stopping=False,
        multiclass=False
        ):
        super().__init__(
            hidden_layers,
            n_iter,
            learning_rate,
            normalize,
            momentum,
            nesterov,
            early_stopping,
            multiclass
            )

    def _softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def _forward_propagation(self, X):
        activations = {'A0': X}

        C = len(self.parameters_) // 2

        for c in range(1, C):
            Z = self.parameters_[f'W{c}'].dot(activations[f'A{c - 1}']) + self.parameters_[f'b{c}']
            activations[f'A{c}'] = 1 / (1 + np.exp(-Z))

        Z = self.parameters_[f'W{C}'].dot(activations[f'A{C - 1}']) + self.parameters_[f'b{C}']
        if self.multiclass_ == True:
            activations[f'A{C}'] = self._softmax(Z)
        else:
            activations[f'A{C}'] = 1 / (1 + np.exp(-Z))

        return activations

    def _backward_propagation(self, y, activations):
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

    def _GD(self, gradients):
        C = len(self.parameters_) // 2
        
        for c in range(1, C + 1):
            self.optimizers_[f'vdW{c}'] = self.momentum_ * self.optimizers_[f'vdW{c}'] + self.learning_rate_ * gradients[f'dW{c}']
            self.optimizers_[f'vdb{c}'] = self.momentum_ * self.optimizers_[f'vdb{c}'] + self.learning_rate_ * gradients[f'db{c}']

            if self.nesterov_ == True:
                self.optimizers_[f'vdW{c}'] = self.momentum_ * self.optimizers_[f'vdW{c}'] + self.learning_rate_ * gradients[f'dW{c}']
                self.optimizers_[f'vdb{c}'] = self.momentum_ * self.optimizers_[f'vdb{c}'] + self.learning_rate_ * gradients[f'db{c}']

            self.parameters_[f'W{c}'] -= self.optimizers_[f'vdW{c}']
            self.parameters_[f'b{c}'] -= self.optimizers_[f'vdb{c}']

    def _Adam(self, gamma, gradients, curr_epoch):
        C = len(self.parameters_) // 2
        
        for c in range(1, C + 1):
            self.optimizers_[f'mdW{c}'] = gamma[0] * self.optimizers_[f'mdW{c}'] + (1 - gamma[0]) * gradients[f'dW{c}']
            self.optimizers_[f'mdb{c}'] = gamma[0] * self.optimizers_[f'mdb{c}'] + (1 - gamma[0]) * gradients[f'db{c}']

            self.optimizers_[f'vdW{c}'] = gamma[1] * self.optimizers_[f'vdW{c}'] + (1 - gamma[1]) * gradients[f'dW{c}']**2
            self.optimizers_[f'vdb{c}'] = gamma[1] * self.optimizers_[f'vdb{c}'] + (1 - gamma[1]) * gradients[f'db{c}']**2


            mdw_corr = self.optimizers_[f'mdW{c}'] / (1 - np.power(gamma[0], curr_epoch + 1))
            mdb_corr = self.optimizers_[f'mdb{c}'] / (1 - np.power(gamma[0], curr_epoch + 1))

            vdw_corr = self.optimizers_[f'vdW{c}'] / (1 - np.power(gamma[1], curr_epoch + 1))
            vdb_corr = self.optimizers_[f'vdb{c}'] / (1 - np.power(gamma[1], curr_epoch + 1))

            self.parameters_[f'W{c}'] -= (self.learning_rate_ / np.sqrt(vdw_corr + 1e-08)) * mdw_corr
            self.parameters_[f'b{c}'] -= (self.learning_rate_ / np.sqrt(vdb_corr + 1e-08)) * mdb_corr

    def _RMSprop(self, beta, gradients):
        C = len(self.parameters_) // 2
        
        for c in range(1, C + 1):
            self.optimizers_[f'vdW{c}'] = beta * self.optimizers_[f'vdW{c}'] + (1 - beta) * gradients[f'dW{c}']**2
            self.optimizers_[f'vdb{c}'] = beta * self.optimizers_[f'vdb{c}'] + (1 - beta) * gradients[f'db{c}']**2

            self.parameters_[f'W{c}'] -= (self.learning_rate_ / np.sqrt(self.optimizers_[f'vdW{c}'] + 1e-08)) * gradients[f'dW{c}']
            self.parameters_[f'b{c}'] -= (self.learning_rate_ / np.sqrt(self.optimizers_[f'vdb{c}'] + 1e-08)) * gradients[f'db{c}']

    def log_loss(self, y, A):
        return 1 / y.shape[1] * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def predict_proba(self, X):
        activations = self._forward_propagation(X)
        C = len(self.parameters_) // 2
        Af = activations[f'A{C}']
        return Af

    def predict(self, X):
        if self.multiclass_ == True:
            return np.argmax(self.predict_proba(X), axis=0)
        return self.predict_proba(X) >= 0.5

    def score(self, X, y):
        if self.multiclass_ == True:
            sum =  np.sum(np.equal(np.argmax(y, axis=0), self.predict(X)))
        else:
            sum = np.sum(np.equal(y, self.predict(X)))
        acc = sum / y.shape[1]
        return acc

    def fit(self, X, y, solver='adam', gamma=(0.9, 0.999), beta=0.9, batch_size=1, test_size=0.2, random_state=None):
        try:
            dimensions = list(self.hidden_layers_)
        except TypeError:
            dimensions = [self.hidden_layers_]
        dimensions.insert(0, X.shape[0])
        dimensions.append(y.shape[0])

        if self.normalize_ == True:
            self.normalize_mean_ = X.mean()
            self.normalize_std_ = X.std()
            X = StandartScaler(X)

        self._X = X
        self._y = y
        self.acc_ = []
        self.loss_ = []
        self.val_loss_ = []
        self._initialisation(dimensions)

        X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=test_size, random_state=random_state)
        X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

        for i in range(self.n_iter_):
            size = 0
            while size < X_train.shape[1]:
                try:
                    tmpX = np.array(X_train[:, size:size+batch_size])
                    tmpY = np.array(y_train[:, size:size+batch_size])
                except:
                    tmpX = np.array(X_train[:, size:])
                    tmpY = np.array(y_train[:, size:])
                size += batch_size
                tmpParameters = self.parameters_

                activations = self._forward_propagation(tmpX)
                gradients = self._backward_propagation(tmpY, activations)
                if solver == 'sgd':
                    self._GD(gradients)
                elif solver == 'adam':
                    self._Adam(gamma, gradients, i)
                elif solver == 'RMSprop':
                    self._RMSprop(beta, gradients)

            if i % 10 == 0:
                val_activations = self._forward_propagation(X_test)
                C = len(self.parameters_) // 2
                self.loss_.append(self.log_loss(tmpY, activations[f'A{C}']))
                self.val_loss_.append(self.log_loss(y_test, val_activations[f'A{C}']))
                self.acc_.append(self.score(X_test, y_test))
                if self.early_stopping_ == True and i // 10 > 10 and self.val_loss_[i // 10] > self.val_loss_[i // 10 - 10]:
                    self.parameters_ = tmpParameters
                    break

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

        XX = np.vstack(tuple(dim)).T

        Z = self.predict(XX)

        Z = Z.reshape((resolution, resolution))

        ax[2].pcolormesh(X0, X1, Z, alpha=0.3, zorder=-1)
        ax[2].contour(X0, X1, Z, colors='green')




    
