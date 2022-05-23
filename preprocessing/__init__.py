"""
The :mod:`AI_learn.preprocessing` module includes scaling, centering,
normalization, binarization methods.
"""

from .scaler import StandartScaler, MinMaxScaler
from .one_hot import one_hot
from .shuffle import shuffle_in_unison