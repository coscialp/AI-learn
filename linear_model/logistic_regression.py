import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ..preprocessing import StandartScaler

class LogisticRegression:
    def __init__(self, n_iter=1000, learning_rate=0.01, normalize=False, multi_class=False):
        self.coef_ = None
        self.bias_ = None
        self.learning_rate_ = learning_rate
        self.n_iter_ = n_iter
        self.loss_ = []
        self.acc_ = []
        self._X = None
        self._y = None
        self.normalize_ = normalize
        self.multi_class_ = multi_class

    def predict_proba(self, X):
        Z = X.dot(self.coef_) + self.bias_
        return  1 / (1 + np.exp(-Z))
    
    def predict(self, X):
        if self.multi_class_ == True:
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            print(self.predict_proba(X).shape)
            return self.predict_proba(X) >= 0.5 

    def log_loss(self, y, A):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def score(self, X, y):
        if self.multi_class_ == True:
            return np.sum(np.equal(np.argmax(y, axis=1), self.predict(X))) / len(y)
        else:
            return np.sum(np.equal(y, self.predict(X))) / len(y)

    def fit(self, X, y, test_size=0.2, random_state=None):
        if self.multi_class_ == True:
            self.coef_ = np.random.randn(X.shape[1], y.shape[1])
        else:
            self.coef_ = np.random.randn(X.shape[1], 1)
        self.bias_ = np.random.randn(1)

        if self.normalize_ == True:
            X = StandartScaler(X)

        self._X = X
        self._y = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        for i in range(self.n_iter_):
            A = self.predict_proba(X_train)

            dW = 1 / len(y_train) * np.dot(X_train.T, A - y_train)
            db = 1 / len(y_train) * np.sum(A - y_train)

            self.coef_ -= self.learning_rate_ * dW
            self.bias_ -= self.learning_rate_ * db

            if i % 10 == 0:
                self.loss_.append(self.log_loss(y_train, A))
                self.acc_.append(self.score(X_test, y_test))

    def display_train(self):
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(self.loss_, label='Loss')
        ax[1].plot(self.acc_, label='Accuracy')
        if self.multi_class_ == True:
            ax[2].scatter(self._X[:, 0], self._X[:, 1], c=np.argmax(self._y, axis=1), s=50)
        else:
             ax[2].scatter(self._X[:, 0], self._X[:, 1], c=self._y, s=50)
        x0_lim = ax[2].get_xlim()
        x1_lim = ax[2].get_ylim()

        resolution = 100

        x0 = np.linspace(x0_lim[0], x0_lim[1], resolution)
        x1 = np.linspace(x1_lim[0], x1_lim[1], resolution)

        X0, X1 = np.meshgrid(x0, x1)

        dim = [X0.ravel()]

        for i in range(1, self._X.shape[1]):
            dim.append(X1.ravel())

        XX = np.vstack(tuple(dim)).T


        Z = self.predict(XX)

        Z = Z.reshape((resolution, resolution))

        ax[2].pcolormesh(X0, X1, Z, alpha=0.3, zorder=-1)
        ax[2].contour(X0, X1, Z, colors='green')


class SGDClassifier(LogisticRegression):
    def __init__(self, n_iter=1000, learning_rate=0.01, normalize=False, multi_class=False):
        super().__init__(n_iter, learning_rate, normalize, multi_class)

    def fit(self, X, y, test_size=0.2, random_state=None):
        if self.multi_class_ == True:
            self.coef_ = np.random.randn(X.shape[1], y.shape[1])
        else:
            self.coef_ = np.random.randn(X.shape[1], 1)
        self.bias_ = np.random.randn(1)

        if self.normalize_ == True:
            X = StandartScaler(X)

        self._X = X
        self._y = y

        lr = self.learning_rate_

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        for i in range(self.n_iter_):
            start = np.random.randint(0, len(y_train) - 1)
            end = np.random.randint(start + 1, len(y_train))
            tmpX = X_train[start:end]
            tmpY = y_train[start:end]

            A = self.predict_proba(tmpX)

            dW = 1 / len(tmpY) * np.dot(tmpX.T, A - tmpY)
            db = 1 / len(tmpY) * np.sum(A - tmpY)

            self.coef_ -= lr * dW
            self.bias_ -= lr * db
            lr /= 1.05

            if i % 10 == 0:
                self.loss_.append(self.log_loss(tmpY, A))
                self.acc_.append(self.score(X_test, y_test))
