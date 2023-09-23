from typing import Any, Dict, Callable, Protocol
from numpy import ndarray, float64

NpArray = ndarray[Any, Any]
EvaluationFunction = Callable[[NpArray, NpArray], float64]


class NamedThing(Protocol):
    @property
    def name(self) -> str:
        pass


def export_named_thing(container: Dict[str, NamedThing]) -> Callable:
    def wrapped_decorator(init: NamedThing) -> NamedThing:
        name = init.name
        container[name] = init
        return init

    return wrapped_decorator
