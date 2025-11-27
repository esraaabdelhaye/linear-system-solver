import timeit
import numpy as np

class iterative_method:
    def __init__(self,X0, A, B, iterations, tol, jacobi: bool):
        self.X = np.array(X0, dtype=float)
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.iterations = iterations
        self.tol = tol
        self.jacobi = jacobi


    def solve(self):
      start_time = timeit.default_timer()
      n = len(self.B)
      for it in range(self.iterations):
          old_x = self.X.copy()

          for i in range(n):
              if self.jacobi:
                  sum1 = np.dot(self.A[i, :i], old_x[:i])
                  sum2 = np.dot(self.A[i, i + 1:], old_x[i + 1:])
              else:
                  # these x values are the NEW updated ones because we’ve already computed them earlier in this iteration.
                  sum1 = np.dot(self.A[i, :i], self.X[:i])
                  # these x values are the OLD ones because we’ve computed in this iteration yet.
                  sum2 = np.dot(self.A[i, i + 1:], old_x[i + 1:])

              self.X[i] = (self.B[i] - sum1 - sum2) / self.A[i, i]

          # absolute relative error
          error = np.max(np.abs((self.X - old_x) / (self.X + 1e-12)))  # Division by zero protection
          if error < self.tol:
              end_time = timeit.default_timer()
              time = end_time - start_time
              return [time,it, self.X]
      return None

