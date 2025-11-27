import timeit
import numpy as np

from methods.AbstractSolver import AbstractSolver


class iterative_method(AbstractSolver):
    def __init__(self, X0, A, B, precision, iterations, tol, jacobi: bool, b):
        super().__init__(A, B, precision)
        self.X = np.array(X0, dtype=float)
        self.iterations = iterations
        self.tol = tol
        self.jacobi = jacobi


    def solve(self):
      start_time = timeit.default_timer()
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
              end_time = timeit.default_timer()
              time = end_time - start_time
              return [time,it, self.X]
      return None

    def dot_with_rounding(self, row, vec, adjust):
        total = 0
        for a, x in zip(row, vec):
            total += adjust(a * x)
        return adjust(total)
