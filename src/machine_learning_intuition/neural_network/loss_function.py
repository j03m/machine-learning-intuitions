from typing import Any, Protocol, Dict
from machine_learning_intuition.types import NpArray, NamedThing, export_named_thing
import numpy as np
from numpy import float64

all_loss_functions: Dict[str, NamedThing] = {}


class Loss(NamedThing):
    def __call__(self, y_true: NpArray, y_pred: NpArray) -> float64:
        pass

    def derivative(self, y_true: NpArray, y_pred: NpArray) -> NpArray:
        pass


@export_named_thing(all_loss_functions)
class MSE(Loss):
    def __call__(self, y_true: NpArray, y_pred: NpArray) -> float64:
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: NpArray, y_pred: NpArray) -> NpArray:
        return -(y_true - y_pred)

    @property
    def name(self):
        return "mse"
