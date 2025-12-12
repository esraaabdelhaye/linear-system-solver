import numpy as np
import math
from methods.AbstractSolver import AbstractSolver

from System.SystemData import SystemData
from typing import Dict, Any


class Cholesky(AbstractSolver):
    """
    Implements Cholesky decomposition for solving symmetric, positive-definite systems.
    Decomposes A into L*L^T where L is lower triangular.

    Supports:
    - Step-by-step tracking for single-step mode (Bonus Feature #1)
    - Note: No scaling (Cholesky assumes positive definite symmetric)

    Requirements:
    - Matrix must be symmetric
    - All eigenvalues must be positive

    Process:
    1. Decompose: A = L*L^T
    2. Forward substitution: Ly = b
    3. Backward substitution: L^T x = y
    """

    def __init__(self, data: SystemData):
        """
        Initialize Cholesky solver.

        Args:
            data: SystemData object with symmetric positive-definite matrix
        """
        super().__init__(data)

    def solve(self) -> Dict[str, Any]:
        """
        Main method: performs Cholesky decomposition, forward/backward substitution, and tracks steps.

        Returns:
            Dictionary with success status, solution vector x, and steps (if tracked)

        Raises:
            ValueError: If matrix is not square, symmetric, or positive definite
        """

        A = np.copy(self.A).astype(float)
        b = np.copy(self.b).astype(float)

        print("Solving: Cholesky Decomposition")

        # Record initial state
        self.add_step({
            "operation": "Initial System",
            "description": "Starting Cholesky decomposition",
            "matrix_A": A.tolist(),
            "vector_b": b.tolist()
        })

        # Check if square
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")

        self.add_step({
            "operation": "Matrix Check",
            "description": "Matrix is square",
            "status": "PASS"
        })

        # Check if symmetric
        if not np.allclose(A, A.T, atol=1e-9):
            raise ValueError("Matrix must be symmetric")

        self.add_step({
            "operation": "Symmetry Check",
            "description": "Matrix is symmetric",
            "status": "PASS"
        })

        # Check positive eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        if np.any(eigenvalues <= 0):
            raise ValueError("Matrix must be positive definite (all eigenvalues > 0)")

        self.add_step({
            "operation": "Positive Definiteness Check",
            "description": f"All {len(eigenvalues)} eigenvalues are positive",
            "eigenvalues": [float(e) for e in eigenvalues],
            "status": "PASS"
        })

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

                    self.add_step({
                        "operation": f"Diagonal Element L[{i}][{j}]",
                        "description": f"L[{i}][{j}] = sqrt(A[{i}][{i}] - sum) = {val:.6f}",
                        "index_i": i,
                        "index_j": j,
                        "value": float(val)
                    })
                else:  # Off-diagonal elements
                    if L[j][j] == 0:
                        raise ValueError("Matrix is not positive definite")
                    val = self.round_sig_fig((A[i][j] - sum_val) / L[j][j])
                    L[i][j] = val

                    self.add_step({
                        "operation": f"Off-diagonal Element L[{i}][{j}]",
                        "description": f"L[{i}][{j}] = (A[{i}][{j}] - sum) / L[{j}][{j}] = {val:.6f}",
                        "index_i": i,
                        "index_j": j,
                        "value": float(val)
                    })

        self.add_step({
            "operation": "Decomposition Complete",
            "description": "L matrix computed: A = L*L^T",
            "L_matrix": L.tolist()
        })

        # Forward Substitution (Ly = b)
        y = np.zeros(self.n)
        for i in range(self.n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            sum_val = self.round_sig_fig(sum_val)
            y[i] = self.round_sig_fig((b[i] - sum_val) / L[i][i])

            self.add_step({
                "operation": f"Forward Substitution: y[{i}]",
                "description": f"Solve Ly=b for position {i}",
                "position": i,
                "value": float(y[i])
            })

        self.add_step({
            "operation": "Forward Substitution Complete",
            "description": "Intermediate vector y computed",
            "y_vector": y.tolist()
        })

        # Backward Substitution (L^T x = y)
        x = np.zeros(self.n)
        for i in reversed(range(self.n)):
            sum_val = sum(L[j][i] * x[j] for j in range(i + 1, self.n))
            sum_val = self.round_sig_fig(sum_val)
            x[i] = self.round_sig_fig((y[i] - sum_val) / L[i][i])

            self.add_step({
                "operation": f"Backward Substitution: x[{i}]",
                "description": f"Solve L^T x=y for position {i}",
                "position": i,
                "value": float(x[i])
            })

        self.add_step({
            "operation": "Solution Found",
            "description": "Backward substitution complete",
            "solution": x.tolist()
        })

        # Return results
        return {
            "success": True,
            "sol": x,
            "steps": self.steps
        }