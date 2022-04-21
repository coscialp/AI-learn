import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, n_iter=1000, learning_rate=0.01, epsillon=0):
        self.coef_ = None
        self.bias_ = None
        self.learning_rate_ = learning_rate
        self.n_iter_ = n_iter
        self.loss_ = []
        self.acc_ = []
        self.epsillon_ = epsillon
        self._X = None
        self._y = None

    def predict(self, X):
        return X.dot(self.coef_) + self.bias_

    def quadratics_loss(self, y, A):
        return 1 / len(y) * np.sum((y - (A + self.epsillon_))**2)

    def score(self, X, y_true):
        y_pred = self.predict(X)
        u = ((y_true - y_pred)**2).sum()
        v = (((y_true - y_true.mean()) ** 2).sum())
        return 1 - (u / v)

    def fit(self, X, y, test_size=0.2,random_state=None, normalize=False):
        self.coef_ = np.random.randn(X.shape[1], 1)
        self.bias_ = np.random.randn(1)

        if normalize == True:
            X = (X - X.mean()) / X.std()
        
        self._X = X
        self._y = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        for i in range(self.n_iter_):
            A = self.predict(X_train)

            dW = 1 / len(y_train) * np.dot(X_train.T, A - y_train)
            db = 1 / len(y_train) * np.sum(A - y_train)

            self.coef_ -= self.learning_rate_ * dW
            self.bias_ -= self.learning_rate_ * db

            if i % 10 == 0:
                self.loss_.append(self.quadratics_loss(y_train, A))
                self.acc_.append(self.score(X_test, y_test))
        print(self.acc_)
    
    def display_train(self):
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(self.loss_, label='Loss')
        ax[1].plot(self.acc_, label='Accuracy')
        ax[2].scatter(self._X[:, 0], self._y[:])
        x0 = np.linspace(ax[2].get_xlim()[0], ax[2].get_xlim()[1])
        x0 = np.reshape(x0, (x0.shape[0], 1))
        ax[2].plot(x0, self.predict(x0), c='red')
        ax[0].legend()
        ax[1].legend()
        plt.show()

