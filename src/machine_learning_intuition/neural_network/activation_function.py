from abc import ABC, abstractmethod
from numpy import ndarray, float64
from machine_learning_intuition.types import NpArray


@abstractmethod
class ActivationFunction(ABC):
    def __call__(self, x: NpArray) -> NpArray:
        raise NotImplementedError()

    def derivative(self, x: NpArray) -> NpArray:
        raise NotImplementedError()


class Linear(ActivationFunction):
    def __call__(self, x: NpArray) -> NpArray:
        return x

    def derivative(self, x: NpArray) -> NpArray:
        return 1
