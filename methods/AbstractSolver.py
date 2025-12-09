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