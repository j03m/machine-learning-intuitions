from machine_learning_intuition.types import NpArray, NamedThing, export_named_thing
import numpy as np
from typing import Protocol, Dict

all_activation_functions: Dict[str, NamedThing] = {}


class ActivationFunction(NamedThing):
    def __call__(self, x: NpArray) -> NpArray:
        pass

    def derivative(self, x: NpArray) -> NpArray:
        pass


@export_named_thing(all_activation_functions)
class Linear(ActivationFunction):
    def __call__(self, x: NpArray) -> NpArray:
        return x

    def derivative(self, x: NpArray) -> NpArray:
        return np.array(1)

    def name(self):
        return "linear"


@export_named_thing(all_activation_functions)
class ReLU(ActivationFunction):
    def __call__(self, x: NpArray) -> NpArray:
        return np.where(x >= 0, x, 0)

    def derivative(self, x: NpArray) -> NpArray:
        return np.where(x >= 0, 1, 0)

    def name(self):
        return "relu"
