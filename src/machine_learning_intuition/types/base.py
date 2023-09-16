from typing import Any, Callable, Protocol
from numpy import ndarray, float64
from typing import NewType

PredictionArray = NewType('PredictionArray', ndarray[Any, Any])
EvaluationFunction = Callable[[PredictionArray, PredictionArray], float64]
