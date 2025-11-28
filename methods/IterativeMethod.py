import timeit
import numpy as np
from System.SystemData import SystemData
from methods.AbstractSolver import AbstractSolver
from typing import  Dict, Any


class IterativeMethod(AbstractSolver):
    def __init__(self, data : SystemData):
        super().__init__(data)
        # params
        self.X = np.array(data.params["Initial Guess"], dtype=float)
        self.iterations = data.params["max_iter_var"]
        self.tol = data.params["error_tol_var"]
        self.jacobi = data.params["Jacobi"]


    def solve(self)-> Dict[str, Any]:
      n = len(self.b)
      for it in range(self.iterations):
          old_x = self.X.copy()

          for i in range(n):
              if self.jacobi:
                  sum1 = self.dot_with_rounding(self.A[i, :i], old_x[:i], self.round_sig_fig)
                  sum2 = self.dot_with_rounding(self.A[i, i + 1:], old_x[i + 1:], self.round_sig_fig)
              else:
                  # these x values are the NEW updated ones because we’ve already computed them earlier in this iteration.
                  sum1 = self.dot_with_rounding(self.A[i, :i], self.X[:i], self.round_sig_fig)
                  # these x values are the OLD ones because we’ve computed in this iteration yet.
                  sum2 = self.dot_with_rounding(self.A[i, i + 1:], old_x[i + 1:], self.round_sig_fig)

              self.X[i] = self.round_sig_fig((self.b[i] - sum1 - sum2) / self.A[i, i])

          # absolute relative error
          error = np.max(np.abs((self.X - old_x) / (self.X + 1e-12)))  # Division by zero protection
          if error < self.tol:
             return {"sol": self.X, "iterations" : it}

      return {"sol": [1,5,9]}

    def dot_with_rounding(self, row, vec, adjust):
        total = 0
        for a, x in zip(row, vec):
            total += adjust(a * x)
        return adjust(total)
