import timeit
from methods.AbstractSolver import AbstractSolver
from System.SystemData import SystemData
from typing import Dict, Any


class Doolittle(AbstractSolver):
    """
    Doolittle LU Decomposition method for solving linear systems.
    Decomposes A into L*U where:
    - L is lower triangular with 1s on diagonal
    - U is upper triangular

    Supports:
    - Scaled partial pivoting (Bonus Feature #3)
    - Step-by-step tracking for single-step mode (Bonus Feature #1)

    Process:
    1. Decompose: A = L*U with pivoting
    2. Forward substitution: Ly = b
    3. Backward substitution: Ux = y
    """

    def __init__(self, data: SystemData):
        """Initialize Doolittle solver"""
        super().__init__(data)

        # Initialize solution vector
        self.x = [0] * self.n

        # Tolerance for detecting zero/near-zero pivots
        self.tol = 1e-9
        self.er = 0
        self.o = list(range(self.n))  # Order vector for row permutations
        self.s = [0] * self.n  # Scaling vector

        # Copy of A (used to store both L and U)
        self.a = self.A.copy()

    def solve(self) -> Dict[str, Any]:
        """
        Solve using Doolittle LU decomposition.

        Returns:
            Dictionary with solution and steps (if tracked)
        """
        print("Solving: Doolittle LU Decomposition")

        # Record initial state
        self.add_step({
            "operation": "Initial System",
            "description": "Starting Doolittle decomposition",
            "matrix_A": self.a.copy().tolist()
        })

        # Doolittle LU decomposition
        self.decompose()
        if self.er == -1:
            raise ValueError("Matrix is singular or near-singular (zero pivot detected)")

        self.add_step({
            "operation": "Decomposition Complete",
            "description": "L*U decomposition finished",
            "matrix_LU": self.a.copy().tolist()
        })

        # Forward + backward substitution
        self.substitute()

        self.add_step({
            "operation": "Solution Found",
            "description": "Forward and backward substitution complete",
            "solution": self.x
        })

        return {
            "success": True,
            "sol": self.x,
            "steps": self.steps
        }

    def decompose(self):
        """
        Perform Doolittle LU decomposition with partial pivoting.
        """
        # Build the scaling vector based on largest absolute value in each row
        for i in range(self.n):
            self.o[i] = i
            self.s[i] = abs(self.a[i, 0])
            for j in range(1, self.n):
                if abs(self.a[i, j]) > self.s[i]:
                    self.s[i] = abs(self.a[i, j])

        self.add_step({
            "operation": "Scaling Factors Computed",
            "description": "Max absolute value per row",
            "scales": [float(s) for s in self.s]
        })

        # Main Doolittle loop for all columns except last
        for k in range(self.n - 1):
            # Perform partial pivoting
            self.pivot(self.a, self.o, self.s, self.n, k)

            # Check for singularity
            if super().round_sig_fig(abs(self.a[self.o[k], k]) / self.s[self.o[k]]) < self.tol:
                self.er = -1
                self.add_step({
                    "operation": "Pivot Check",
                    "description": f"Zero or near-zero pivot at column {k}",
                    "status": "FAILED"
                })
                return

            self.add_step({
                "operation": f"Decompose Column {k}",
                "description": f"Compute L and U for column {k}",
                "column": k,
                "pivot_row": self.o[k],
                "matrix_state": self.a.copy().tolist()
            })

            # Eliminate entries below pivot
            for i in range(k + 1, self.n):
                factor = super().round_sig_fig(self.a[self.o[i], k] / self.a[self.o[k], k])
                # Store L(i,k) in A (lower-triangular)
                self.a[self.o[i], k] = factor

                # Update remaining elements in the row (U part)
                for j in range(k + 1, self.n):
                    self.a[self.o[i], j] = super().round_sig_fig(self.a[self.o[i], j] - factor * self.a[self.o[k], j])

        # Check final pivot
        if super().round_sig_fig(abs(self.a[self.o[self.n - 1], self.n - 1])) < self.tol:
            self.er = -1
            self.add_step({
                "operation": "Final Pivot Check",
                "description": "Zero pivot in final position",
                "status": "FAILED"
            })

    def substitute(self):
        """
        Forward and backward substitution to solve Ly=b and Ux=y.
        """
        a, o, n, b, x = self.a, self.o, self.n, self.b, self.x
        y = [0] * n

        # Forward substitution to solve Ly = b
        y[o[0]] = b[o[0]]
        for i in range(1, n):
            sum_val = b[o[i]]
            for j in range(i):
                sum_val = super().round_sig_fig(sum_val - a[o[i], j] * y[o[j]])
            y[o[i]] = sum_val

            self.add_step({
                "operation": f"Forward Substitution: y[{i}]",
                "description": f"Solve Ly=b for position {i}",
                "position": i,
                "value": float(y[o[i]])
            })

        # Backward substitution to solve Ux = y
        x[n - 1] = super().round_sig_fig(y[o[n - 1]] / a[o[n - 1], n - 1])
        for i in range(n - 2, -1, -1):
            sum_val = 0
            for j in range(i + 1, n):
                sum_val = super().round_sig_fig(sum_val + a[o[i], j] * x[j])
            x[i] = super().round_sig_fig((y[o[i]] - sum_val) / a[o[i], i])

            self.add_step({
                "operation": f"Backward Substitution: x[{i}]",
                "description": f"Solve Ux=y for position {i}",
                "position": i,
                "value": float(x[i])
            })

    def pivot(self, a, o, s, n, k):
        """
        Find the largest scaled coefficient in a column and perform pivoting.

        Args:
            a: Matrix
            o: Order vector
            s: Scaling vector
            n: Matrix size
            k: Column index
        """
        p = k
        big = super().round_sig_fig(abs(a[o[k], k]) / s[o[k]])
        for i in range(k + 1, n):
            dummy = super().round_sig_fig(abs(a[o[i], k] / s[o[i]]))
            if dummy > big:
                big = dummy
                p = i

        # Swap the order of pivot row and original row
        dummy = o[p]
        o[p] = o[k]
        o[k] = dummy

        if p != k:
            self.add_step({
                "operation": f"Pivot at Column {k}",
                "description": f"Swap rows {o[k]} and {o[p]} for largest scaled element",
                "column": k,
                "from_row": o[p],
                "to_row": o[k]
            })