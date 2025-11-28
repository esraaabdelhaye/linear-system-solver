import numpy as np
import math
from methods.AbstractSolver import AbstractSolver

from System.SystemData import SystemData
from typing import  Dict, Any

class Cholesky(AbstractSolver):
    """
    Implements Cholesky decomposition for solving symmetric, positive-definite systems.
    Supports optional row scaling and logs all steps.
    """

    def __init__(self,data: SystemData):
        super().__init__(data)

        # Compute scales if scaling is used, otherwise use ones

    def solve(self)-> Dict[str, Any]:

        """
        Main method: performs Cholesky decomposition, forward/backward substitution, and logs steps.
        Returns a dictionary containing solution, L matrix, execution time, and steps.
        """

        A = np.copy(self.A).astype(float)
        b = np.copy(self.b).astype(float)

<<<<<<< HEAD
        # validation
=======
        # validation 
>>>>>>> 571a1a5da18de8d69472be13a29473d74380b032
        # Check if square
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")
        # Check if symmetric
        if not np.allclose(A, A.T, atol=1e-9):
            raise ValueError("Matrix must be symmetric")
        # Check positive eigenvalues
        if np.any(np.linalg.eigvals(A) <= 0):
            raise ValueError("Matrix must be positive definite (all eigenvalues > 0)")

<<<<<<< HEAD
=======
        
>>>>>>> 571a1a5da18de8d69472be13a29473d74380b032
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
        # self.add_step("L matrix", L.tolist())

        
        # Forward Substitution (Ly = b)
        y = np.zeros(self.n)
        for i in range(self.n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            sum_val = self.round_sig_fig(sum_val)
            y[i] = self.round_sig_fig((b[i] - sum_val) / L[i][i])
        # self.add_step("Forward substitution (Y)", y.tolist())

        
        # Backward Substitution (L^T x = y)
        x = np.zeros(self.n)
        for i in reversed(range(self.n)):
            sum_val = sum(L[j][i] * x[j] for j in range(i + 1, self.n))
            sum_val = self.round_sig_fig(sum_val)
            x[i] = self.round_sig_fig((y[i] - sum_val) / L[i][i])
        # self.add_step("Backward substitution (X)", x.tolist())

        


        # Return results
        return {
            # "errorMessage": "",
            "sol": x
            # "L_matrix": L
        }

