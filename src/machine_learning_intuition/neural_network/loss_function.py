from typing import Any
from abc import ABC, abstractmethod
from machine_learning_intuition.types import PredictionArray
import numpy as np

__all__ = ['Loss', 'MSE']


@abstractmethod
class Loss(ABC):
    def loss(self, y_true: PredictionArray, y_pred: PredictionArray) -> Any:
        raise NotImplementedError()

    def gradient(self, y_true: PredictionArray, y_pred: PredictionArray) -> Any:
        raise NotImplementedError()


class MSE(Loss):
    def loss(self, y_true: PredictionArray, y_pred: PredictionArray) -> Any:
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true: PredictionArray, y_pred: PredictionArray) -> Any:
        return 2 * (y_pred - y_true)
