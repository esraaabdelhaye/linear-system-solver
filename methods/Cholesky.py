import numpy as np
import math
import time
from AbstractSolver import AbstractSolver

class Cholesky(AbstractSolver):
    """
    Implements Cholesky decomposition for solving symmetric, positive-definite systems.
    Supports optional row scaling and logs all steps.
    """

    def __init__(self, A, b, precision=6, single_step=False, use_scaling=False):
        super().__init__(A, b, precision, single_step)
        self.use_scaling = False        # applying scaling in cholesky breaks symmetry --> disabled

        # Compute scales if scaling is used, otherwise use ones

    def solve(self):
        """
        Main method: performs Cholesky decomposition, forward/backward substitution, and logs steps.
        Returns a dictionary containing solution, L matrix, execution time, and steps.
        """
        self.validate()
        start_time = time.time()  # Start timing
        A = np.copy(self.A).astype(float)
        b = np.copy(self.b).astype(float)

        
        # Cholesky Decomposition (A = L * L^T)
        L = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1):
                # Compute sum for previous L elements
                sum_val = sum(L[i][k] * L[j][k] for k in range(j))
                sum_val = self.round_sig_fig(sum_val)

                if i == j:  # Diagonal elements
                    val = self.round_sig_fig(math.sqrt(A[i][i] - sum_val))
                    L[i][j] = val
                else:       # Off-diagonal elements
                    if L[j][j] == 0:
                        raise ValueError("Matrix is not positive definite")
                    val = self.round_sig_fig((A[i][j] - sum_val) / L[j][j])
                    L[i][j] = val
        self.add_step("L matrix", L.tolist())

        
        # Forward Substitution (Ly = b)
        y = np.zeros(self.n)
        for i in range(self.n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            sum_val = self.round_sig_fig(sum_val)
            y[i] = self.round_sig_fig((b[i] - sum_val) / L[i][i])
        self.add_step("Forward substitution (Y)", y.tolist())

        
        # Backward Substitution (L^T x = y)
        x = np.zeros(self.n)
        for i in reversed(range(self.n)):
            sum_val = sum(L[j][i] * x[j] for j in range(i + 1, self.n))
            sum_val = self.round_sig_fig(sum_val)
            x[i] = self.round_sig_fig((y[i] - sum_val) / L[i][i])
        self.add_step("Backward substitution (X)", x.tolist())

        
        # Execution Time
        exec_time = time.time() - start_time
        self.add_step("Computation Time", exec_time)

        # Return results
        return {
            "solution": x,
            "L_matrix": L,
            "execution_time": exec_time,
            "steps": self.steps
        }

