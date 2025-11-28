import math
from copy import deepcopy

import numpy as np
from typing import  Dict, Any
from System.SystemData import SystemData


class AbstractSolver:
    def __init__(self, data: SystemData):
        # Store matrix A and vector b as NumPy arrays for numerical operations
        self.A = np.array(data.A)
        self.b = np.array(data.b)
        self.precision = data.precision     # Precision for rounding (significant figures)
        self.n = data.N                    # Number of variables / matrix size
        self.steps = []                   # List to store step-by-step logs (used in GUI for explanations)


    def solve(self)  -> Dict[str, Any]:
        """
        Virtual method to be overridden by child solvers.
        Each specific solver implements its own solve().
        """
        pass
    
    
    def round_sig_fig(self, x, n=None):
        """
        Rounds a number x to n significant figures.
        If n is not provided, uses the solver's precision value.
        """
        if n is None:
            n = self.precision
        if x == 0:
            return 0.0
        
        # Formula for rounding to significant figures: round(x, n - 1 - floor(log10(|x|)))
        return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)

    def get_scales(self):
        """
        Computes row scaling factors used for scaled partial pivoting.
        For each row i:
            scale[i] = max(|A[i][j]|) over all j
        If the row is all zeros, scale = 1 to avoid division by zero.
        """
        scales = []
        for i in range(self.n):
            # Get maximum absolute value in row i
            max_val = max(abs(val) for val in self.A[i])

            # Append scale (use 1 if row is all zeros)
            scales.append(max_val if max_val != 0 else 1)
        return scales

    def add_step(self, step):
        """
        Adds a step to the step log.
        Used for GUI-based step-by-step visualizations.
        """
        self.steps.append(step)
