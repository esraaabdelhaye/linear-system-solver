import math
from typing import Dict, Any
from abc import ABC, abstractmethod


class AbstractRootFinder(ABC):
    """Abstract base class for all root-finding methods"""

    def __init__(self, data):
        self.equation = data.equation
        self.precision = data.precision
        self.params = data.params
        self.steps = []

    def round_sig_fig(self, x: float, n: int = None) -> float:
        if n is None:
            n = self.precision
        if x == 0:
            return 0.0
        return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)

    def evaluate(self, x: float) -> float:
        try:
            safe_expr = (self.equation
                         .replace('^', '**')
                         .replace('sin', 'math.sin')
                         .replace('cos', 'math.cos')
                         .replace('exp', 'math.exp')
                         .replace('sqrt', 'math.sqrt')
                         .replace('log', 'math.log')
                         .replace('tan', 'math.tan')
                         .replace('abs', 'math.fabs'))
            return eval(safe_expr, {"x": x, "math": math, "__builtins__": {}})
        except Exception as e:
            raise ValueError(f"Error evaluating equation '{self.equation}' at x={x}: {e}")

    def numerical_derivative(self, x: float, h: float = 1e-7) -> float:
        try:
            return (self.evaluate(x + h) - self.evaluate(x - h)) / (2 * h)
        except:
            return None

    def _count_correct_sig_figs(self, current: float, previous: float) -> int:
        if current == 0 or previous == 0:
            return 0
        try:
            error = abs((current - previous) / (current if current != 0 else 1))
            if error == 0:
                return self.precision
            return max(0, -int(math.floor(math.log10(error))))
        except:
            return 0

    def add_step(self, step: Dict[str, Any]):
        self.steps.append(step)

    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        pass