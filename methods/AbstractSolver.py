import math
from copy import deepcopy

import numpy as np
from typing import Dict, Any
from System.SystemData import SystemData


class AbstractSolver:
    """
    Abstract base class for all linear system solvers.

    Features:
    - Precision control (significant figures rounding)
    - Scaling support for partial pivoting
    - Step-by-step tracking for single-step mode (Bonus #1)
    """

    def __init__(self, data: SystemData):
        self.A = np.array(data.A)
        self.b = np.array(data.b)
        self.precision = data.precision
        self.n = data.N
        self.steps = []  # For single-step mode bonus
        self.use_scaling = data.use_scaling  # Bonus #3: scaling flag

    def solve(self) -> Dict[str, Any]:
        """Solve the system and return results"""
        pass

    def round_sig_fig(self, x, n=None):
        """
        Round to specified number of significant figures.

        Args:
            x: Number to round
            n: Number of significant figures (uses self.precision if None)

        Returns:
            Rounded number
        """
        if n is None:
            n = self.precision
        if x == 0:
            return 0.0
        return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)

    def get_scales(self):
        """
        Calculate scaling factors for each row (max absolute value per row).
        Used for scaled partial pivoting (Bonus #3).

        Returns:
            List of scaling factors
        """
        scales = []
        for i in range(self.n):
            max_val = max(abs(val) for val in self.A[i])
            scales.append(max_val if max_val != 0 else 1)
        return scales

    def add_step(self, step):
        """
        Record a step for single-step mode (Bonus #1).

        Args:
            step: Dictionary or string describing the step performed
                  Can contain: operation, row info, matrix state, etc.
        """
        self.steps.append(step)