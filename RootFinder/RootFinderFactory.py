from typing import Dict, Any

from .ModifiedSecant import ModifiedSecant
from .RootFinderData import RootFinderData
from .Bisection import Bisection
from .FixedPoint import FixedPoint
from .FalsePosition import FalsePosition
from .NewtonRaphson import NewtonRaphson
from .Modified1NewtonRaphson import Modified1NewtonRaphson
from .Modified2NewtonRaphson import Modified2NewtonRaphson
from .Secant import Secant


# from .FalsePosition import FalsePosition
# from .NewtonRaphson import NewtonRaphson
# from .ModifiedNewtonRaphson import ModifiedNewtonRaphson
# from .Secant import Secant


class RootFinderFactory:
    """Factory Pattern Implementation for Root Finders"""

    METHODS = {
        "Bisection": Bisection,
        "False-Position": FalsePosition,
        "Fixed Point": FixedPoint,
        "Newton-Raphson": NewtonRaphson,
        "Modified Newton-Raphson (Known m)": Modified1NewtonRaphson,
        "Modified Newton-Raphson (Unknown m)":  Modified2NewtonRaphson,
        "Secant": Secant,
        "Modified Secant": ModifiedSecant
    }

    @staticmethod
    def get_solver(data: RootFinderData):
        """Factory method to instantiate the correct root finder"""
        solver_class = RootFinderFactory.METHODS.get(data.method)

        if not solver_class:
            raise ValueError(
                f"Root finder method '{data.method}' is not implemented.\n"
                f"Available methods: {', '.join(RootFinderFactory.METHODS.keys())}"
            )

        return solver_class(data)