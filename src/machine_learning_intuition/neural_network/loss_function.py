from typing import Any
from abc import ABC, abstractmethod
from machine_learning_intuition.types import NpArray
import numpy as np
from numpy import float64

@abstractmethod
class Loss(ABC):
    def __call__(self, y_true: NpArray, y_pred: NpArray) -> float64:
        raise NotImplementedError()

    def derivative(self, y_true: NpArray, y_pred: NpArray) -> NpArray:
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()


class MSE(Loss):
    def __call__(self, y_true: NpArray, y_pred: NpArray) -> float64:
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: NpArray, y_pred: NpArray) -> NpArray:
        return 2 * (y_pred - y_true)

    def name(self):
        return "mse"


all_loss_functions = {
    "mse": MSE
}
