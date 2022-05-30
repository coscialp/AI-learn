"""
Multi-layer Perceptron
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ..preprocessing import shuffle_in_unison, StandartScaler, MinMaxScaler

ACTIVATION = {
    'relu': lambda Z: np.where(Z <= 0, 0, Z),
    'logistic': lambda Z: 1 / (1 + np.exp(-Z)),
    'tanh': lambda Z: np.tanh(Z),
    'identity': lambda Z: Z,
    'softmax': lambda Z: np.exp(Z) / np.sum(np.exp(Z), axis=0)
}

DERIVATIVES = {
    'relu': lambda delta, Z: delta * np.where(Z <= 0, 0, 1),
    'logistic': lambda delta, Z: delta * Z * (1 - Z),
    'tanh': lambda delta, Z: delta * (1 - Z**2),
    'identity': lambda delta, Z: delta,
}

class BaseMLP:

    def __init__(
        self,
        hidden_layers=(32, 32, 32),
        n_iter=1000,
        learning_rate_init=0.01,
        learning_rate='constant',
        normalize=False,
        momentum=0.9,
        nesterov=True,
        early_stopping=False,
        multiclass=False,
        shuffle=True,
        activation='relu',
        out_activation='logistic',
        solver='adam',
        batch_size=16,
        gamma=(0.9, 0.999),
        beta=0.9,
        epsilon=1e-08
        ):
        self.parameters_ = {}
        self.optimizers_ = {}
        self.n_iter_ = n_iter
        self.learning_rate_ = learning_rate_init
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
        self.evoluate_lr_ = learning_rate
        self.shuffle_ = shuffle
        self.activation_ = activation
        self.out_activation_ = out_activation
        self.solver_ = solver
        self.batch_size_ = batch_size
        self.beta_ = beta
        self.gamma_ = gamma
        self.curr_epoch_ = 0
        self.epsilon_ = epsilon


    def _initialisation(self, dimensions):
        C = len(dimensions)

        factor = 6.0 if self.activation_ != 'logistic' else 2.0

        for c in range(1, C):
            init_bound = (factor / (dimensions[c - 1] + dimensions[c])) ** 0.5
            # self.parameters_[f'W{c}'] = np.random.uniform(-1 / (dimensions[c - 1] ** 0.5), 1 / (dimensions[c - 1] ** 0.5), (dimensions[c], dimensions[c - 1]))
            # self.parameters_[f'b{c}'] = np.random.uniform(-1 / (dimensions[c - 1] ** 0.5), 1 / (dimensions[c - 1] ** 0.5), (dimensions[c], 1))
            self.parameters_[f'W{c}'] = np.random.uniform(-init_bound, init_bound, (dimensions[c], dimensions[c - 1]))
            self.parameters_[f'b{c}'] = np.random.uniform(-init_bound, init_bound, (dimensions[c], 1))
            self.optimizers_[f'mdW{c}'] = 0
            self.optimizers_[f'mdb{c}'] = 0
            self.optimizers_[f'vdW{c}'] = 0
            self.optimizers_[f'vdb{c}'] = 0


class MLPClassifier(BaseMLP):
    """
    
    """
    def __init__(
        self,
        hidden_layers=(32, 32, 32),
        n_iter=1000,
        learning_rate_init=0.01,
        learning_rate='constant',
        normalize=False,
        momentum=0.9,
        nesterov=True,
        early_stopping=False,
        multiclass=False,
        shuffle=True,
        activation='relu',
        out_activation='sigmoid',
        solver='adam',
        batch_size=16,
        gamma=(0.9, 0.999),
        beta=0.9,
        epsilon=1e-08
        ):

        self.SOLVER = {
            'adam': self._Adam,
            'sgd': self._GD,
            'RMSprop': self._RMSprop
        }

        super().__init__(
            hidden_layers,
            n_iter,
            learning_rate_init,
            learning_rate,
            normalize,
            momentum,
            nesterov,
            early_stopping,
            multiclass,
            shuffle,
            activation,
            out_activation,
            solver,
            batch_size,
            gamma,
            beta,
            epsilon
            )

    def _forward_propagation(self, X):
        activations = {'A0': X}

        C = len(self.parameters_) // 2

        for c in range(1, C):
            Z = self.parameters_[f'W{c}'].dot(activations[f'A{c - 1}']) + self.parameters_[f'b{c}']
            activations[f'A{c}'] = ACTIVATION[self.activation_](Z)

        Z = self.parameters_[f'W{C}'].dot(activations[f'A{C - 1}']) + self.parameters_[f'b{C}']
        activations[f'A{C}'] = ACTIVATION[self.out_activation_](Z)

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
                dZ = DERIVATIVES[self.activation_](np.dot(self.parameters_[f'W{c}'].T, dZ), activations[f'A{c - 1}'])

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

    def _Adam(self, gradients):
        C = len(self.parameters_) // 2
        
        for c in range(1, C + 1):
            self.optimizers_[f'mdW{c}'] = self.gamma_[0] * self.optimizers_[f'mdW{c}'] + (1 - self.gamma_[0]) * gradients[f'dW{c}']
            self.optimizers_[f'mdb{c}'] = self.gamma_[0] * self.optimizers_[f'mdb{c}'] + (1 - self.gamma_[0]) * gradients[f'db{c}']

            self.optimizers_[f'vdW{c}'] = self.gamma_[1] * self.optimizers_[f'vdW{c}'] + (1 - self.gamma_[1]) * gradients[f'dW{c}']**2
            self.optimizers_[f'vdb{c}'] = self.gamma_[1] * self.optimizers_[f'vdb{c}'] + (1 - self.gamma_[1]) * gradients[f'db{c}']**2


            mdw_corr = self.optimizers_[f'mdW{c}'] / (1 - np.power(self.gamma_[0], self.curr_epoch_ + 1))
            mdb_corr = self.optimizers_[f'mdb{c}'] / (1 - np.power(self.gamma_[0], self.curr_epoch_ + 1))

            vdw_corr = self.optimizers_[f'vdW{c}'] / (1 - np.power(self.gamma_[1], self.curr_epoch_ + 1))
            vdb_corr = self.optimizers_[f'vdb{c}'] / (1 - np.power(self.gamma_[1], self.curr_epoch_ + 1))

            self.parameters_[f'W{c}'] -= (self.learning_rate_ / np.sqrt(vdw_corr + self.epsilon_)) * mdw_corr
            self.parameters_[f'b{c}'] -= (self.learning_rate_ / np.sqrt(vdb_corr + self.epsilon_)) * mdb_corr

    def _RMSprop(self, gradients):
        C = len(self.parameters_) // 2
        
        for c in range(1, C + 1):
            self.optimizers_[f'vdW{c}'] = self.beta_ * self.optimizers_[f'vdW{c}'] + (1 - self.beta_) * gradients[f'dW{c}']**2
            self.optimizers_[f'vdb{c}'] = self.beta_ * self.optimizers_[f'vdb{c}'] + (1 - self.beta_) * gradients[f'db{c}']**2

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

    def fit(
        self,
        X,
        y,
        test_size=0.2,
        random_state=None,
        ):
        try:
            dimensions = list(self.hidden_layers_)
        except TypeError:
            dimensions = [self.hidden_layers_]
        dimensions.insert(0, X.shape[0])
        dimensions.append(y.shape[0])

        if self.normalize_ == True:
            self.normalize_mean_ = X.mean(axis=1)
            self.normalize_std_ = X.std(axis=1)
            X = StandartScaler(X)

            # self.normalize_mean_ = X.min(axis=1)
            # self.normalize_std_ = X.max(axis=1)
            # X = MinMaxScaler(X)

        self._X = X
        self._y = y
        self.acc_ = []
        self.loss_ = []
        self.val_loss_ = []
        self._initialisation(dimensions)

        X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=test_size, random_state=random_state)
        X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

        for i in range(self.n_iter_):
            self.curr_epoch = i
            size = 0
            if self.shuffle_:
                X_train, y_train = shuffle_in_unison(X_train, y_train)
                X_train, y_train = X_train.T, y_train.T
            while size < X_train.shape[1]:
                try:
                    tmpX = np.array(X_train[:, size:size + self.batch_size_])
                    tmpY = np.array(y_train[:, size:size + self.batch_size_])
                except:
                    tmpX = np.array(X_train[:, size:])
                    tmpY = np.array(y_train[:, size:])
                size += self.batch_size_
                tmpParameters = self.parameters_

                activations = self._forward_propagation(tmpX)
                gradients = self._backward_propagation(tmpY, activations)
                self.SOLVER[self.solver_](gradients)

            if  self.solver_ == 'sgd' and self.evoluate_lr_ == 'invscaling':
                self.learning_rate_ /= 1.01

            val_activations = self._forward_propagation(X_test)
            C = len(self.parameters_) // 2
            self.loss_.append(self.log_loss(tmpY, activations[f'A{C}']))
            self.val_loss_.append(self.log_loss(y_test, val_activations[f'A{C}']))
            self.acc_.append(self.score(X_test, y_test))
            if self.early_stopping_ == True and i > 10 and self.val_loss_[i] > self.val_loss_[i - 10]:
                self.parameters_ = tmpParameters
                break


    
