"""
The :mod:`AI_learn.linear_model` module implements a variety of linear models.
"""

from .type import CLASSIFIER, REGRESSOR
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression, SGDClassifier

__all__ = ['CLASSIFIER', 'REGRESSOR', 'LinearRegression', 'LogisticRegression', 'SGDClassifier']