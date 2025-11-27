import math

import numpy as np

class AbstractSolver:
    def __init__(self, A, b, precision = 6, single_step = False):
        self.A = np.array(A)
        self.b = np.array(b)
        self.precision = precision
        self.single_step = single_step
        self.n = len(self.A)

        self.steps = []

        def solve(self):
            pass

        def validate(self):
            if any(len(row) != self.n for row in self.A):
                raise ValueError("Matrix must be square")
            if len(self.A) != self.b:
                raise ValueError("b's size must be equal to A's size")

        def round_sig_fig(x, n):
            if x == 0:
                return '0'
            return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)

        def get_scales(self):
            scales = []
            for i in range(len(self.n)):
                max_val = max(abs(val) for val in self.A[i])
                scales.append(max_val if max_val != 0 else 1)
            return scales

        def add_step(self, step):
            self.steps.append(step)

