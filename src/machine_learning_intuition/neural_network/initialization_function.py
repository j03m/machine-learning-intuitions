from typing import Protocol, Dict
from machine_learning_intuition.types import NpArray, NamedThing, export_named_thing
import numpy as np

all_init_functions: Dict[str, NamedThing] = {}


class InitFunction(NamedThing):
    def __call__(self, input_units: int, output_units: int) -> NpArray:
        pass


@export_named_thing(all_init_functions)
class XavierGlordot(InitFunction):
    def __call__(self, input_units: int, output_units: int) -> NpArray:
        limit = np.sqrt(6 / (input_units + output_units))
        weights_matrix = np.random.uniform(-limit, limit, (input_units, output_units))
        return weights_matrix

    @property
    def name(self):
        return "xavier_glordot"


@export_named_thing(all_init_functions)
class Zeros(InitFunction):
    def __call__(self, input_units: int, output_units: int) -> NpArray:
        return np.zeros((input_units, output_units))

    @property
    def name(self):
        return "zeros"


@export_named_thing(all_init_functions)
class He(InitFunction):
    def __call__(self, input_units: int, output_units: int) -> NpArray:
        limit = np.sqrt(2 / input_units)
        return np.random.uniform(-limit, limit, (input_units, output_units))

    @property
    def name(self):
        return "he"


@export_named_thing(all_init_functions)
class Lecun(InitFunction):
    def __call__(self, input_units: int, output_units: int) -> NpArray:
        limit = np.sqrt(1 / input_units)
        return np.random.uniform(-limit, limit, (input_units, output_units))

    @property
    def name(self):
        return "lecun"

@export_named_thing(all_init_functions)
class Random(InitFunction):
    def __call__(self, input_units: int, output_units: int) -> NpArray:
        return np.random.randn(input_units, output_units)

    @property
    def name(self):
        return "lecun"
