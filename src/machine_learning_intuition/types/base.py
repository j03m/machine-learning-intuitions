from typing import Any, Callable, Protocol
from numpy import ndarray, float64
from typing import NewType

NpArray = ndarray[Any, Any]
EvaluationFunction = Callable[[NpArray, NpArray], float64]

