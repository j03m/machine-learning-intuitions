from abc import ABC, abstractmethod
from machine_learning_intuition.types import NpArray


@abstractmethod
class ActivationFunction(ABC):
    def __call__(self, x: NpArray) -> NpArray:
        raise NotImplementedError()

    def derivative(self, x: NpArray) -> NpArray:
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()


class Linear(ActivationFunction):
    def __call__(self, x: NpArray) -> NpArray:
        return x

    def derivative(self, x: NpArray) -> NpArray:
        return 1

    def name(self):
        return "linear"


all_activations = {
    "linear": Linear
}
