import math
from copy import deepcopy

import numpy as np
from typing import  Dict, Any
from System.SystemData import SystemData


class AbstractSolver:
    def __init__(self, data: SystemData):
        self.A = np.array(data.A)
        self.b = np.array(data.b)
        self.precision = data.precision
        self.n = data.N
        self.steps = []


    def solve(self)  -> Dict[str, Any]:
        pass
    #
    # def validate(self) :
    #     A = deepcopy(self.A)
    #     b = deepcopy(self.b)
    #     n = self.N
    #     for k in range(n):
    #         # Find the row with the largest absolute value in column k
    #         pivot_row = k
    #         max_val = abs(A[k, k])
    #         for i in range(k + 1, n):
    #             if abs(A[i, k]) > max_val:
    #                 max_val = abs(A[i, k])
    #                 pivot_row = i
    #
    #         # If the pivot is almost zero, skip this column
    #         if max_val < 1e-12:
    #             continue
    #
    #         # Swap current row with pivot_row if needed
    #         if pivot_row != k:
    #             A[[k, pivot_row]] = A[[pivot_row, k]]
    #             b[[k, pivot_row]] = b[[pivot_row, k]]
    #
    #         # Eliminate entries below pivot
    #         for i in range(k + 1, n):
    #             factor = A[i, k] / A[k, k]
    #             A[i, k:] -= factor * A[k, k:]
    #             A[i, k] = 0.
    #             b[i] -= factor * b[k]
    #
    #     for i in range(n):
    #         if np.all(np.abs(A[i]) < 1e-12) and abs(b[i]) > 1e-12:
    #             # return {"Success": False, "errorMessage": "System is inconsistent"}
    #             raise ValueError(f"System is inconsistent (row {i} is all zeros in A but b != 0)")
    #         if np.all(np.abs(A[i]) < 1e-12) and abs(b[i]) < 1e-12:
    #             # Row is dependent -> may have infinitely many solutions
    #             # return {"Success": False,"errorMessage": "System has infinite number of solutions"}
    #             raise ValueError(f"System has infinite number of solutions")

    def round_sig_fig(self, x, n=None):
        if n is None:
            n = self.precision
        if x == 0:
            return 0.0
        return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)

    def get_scales(self):
        scales = []
        for i in range(self.n):
            max_val = max(abs(val) for val in self.A[i])
            scales.append(max_val if max_val != 0 else 1)
        return scales

    def add_step(self, step):
        self.steps.append(step)