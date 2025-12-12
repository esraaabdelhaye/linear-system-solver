from methods.AbstractSolver import AbstractSolver
from System.SystemData import SystemData
from typing import Dict, Any


class Crout(AbstractSolver):
    """
    Crout LU Decomposition method for solving linear systems.
    Decomposes A into L*U where:
    - L is lower triangular
    - U is upper triangular with 1s on diagonal

    Supports:
    - Scaled partial pivoting (Bonus Feature #3)
    - Step-by-step tracking for single-step mode (Bonus Feature #1)

    Process:
    1. Decompose: A = L*U with pivoting (column-wise for L)
    2. Forward substitution: Ly = b
    3. Backward substitution: Ux = y
    """

    def __init__(self, data: SystemData):
        """Initialize Crout solver"""
        super().__init__(data)

        # Initialize solution vector
        self.x = [0] * self.n

        # Numerical tolerance for detecting zero pivots
        self.tol = 1e-9

        # Error flag (-1 means failure during decomposition)
        self.er = 0

        # Order vector for row permutations
        self.o = list(range(self.n))

        # Scaling vector for scaled partial pivoting
        self.s = [0] * self.n

        # Copy of matrix A (decomposition happens here)
        self.a = self.A.copy()

    def solve(self) -> Dict[str, Any]:
        """
        Solve using Crout LU decomposition.

        Returns:
            Dictionary with solution and steps (if tracked)
        """
        print("Solving: Crout LU Decomposition")

        # Record initial state
        self.add_step({
            "operation": "Initial System",
            "description": "Starting Crout decomposition",
            "matrix_A": self.a.copy().tolist()
        })

        # Perform LU decomposition (Crout method)
        self.decompose()

        # If decomposition failed, exit
        if self.er == -1:
            raise ValueError("Matrix is singular or near-singular (zero pivot detected)")

        self.add_step({
            "operation": "Decomposition Complete",
            "description": "L*U decomposition finished",
            "matrix_LU": self.a.copy().tolist()
        })

        # Substitute with L and U to get y then x (solution vector)
        self.substitute()

        # Record final solution
        self.add_step({
            "operation": "Solution Found",
            "description": "Forward and backward substitution complete",
            "solution": self.x
        })

        # Return solution vector
        return {
            "success": True,
            "sol": self.x,
            "steps": self.steps
        }

    def decompose(self):
        """
        Perform Crout LU decomposition with partial pivoting.
        """
        # Compute scaling vector used in scaled partial pivoting
        for i in range(self.n):
            self.o[i] = i  # Initial ordering
            self.s[i] = abs(self.a[i, 0])  # Find max absolute element in row
            for j in range(1, self.n):
                if abs(self.a[i, j]) > self.s[i]:
                    self.s[i] = abs(self.a[i, j])

        self.add_step({
            "operation": "Scaling Factors Computed",
            "description": "Max absolute value per row",
            "scales": [float(s) for s in self.s]
        })

        # Begin Crout decomposition: build L column-wise
        for j in range(self.n):  # For each column j

            # Pivoting (on column j)
            self.pivot(self.a, self.o, self.s, self.n, j)

            # Check if scaled pivot is less than tolerance
            if super().round_sig_fig(abs(self.a[self.o[j], j]) / self.s[self.o[j]]) < self.tol:
                self.er = -1
                self.add_step({
                    "operation": "Pivot Check",
                    "description": f"Zero or near-zero pivot at column {j}",
                    "status": "FAILED"
                })
                return

            # Calculate L(i, j) - lower triangular part
            for i in range(j, self.n):
                sum_val = self.a[self.o[i], j]
                for k in range(j):
                    sum_val = super().round_sig_fig(sum_val - self.a[self.o[i], k] * self.a[self.o[k], j])
                self.a[self.o[i], j] = sum_val  # L(i, j)

            self.add_step({
                "operation": f"Compute L Column {j}",
                "description": f"Calculate lower triangular elements for column {j}",
                "column": j,
                "pivot_row": self.o[j]
            })

            # Check zero pivot after building L(j,j)
            if abs(self.a[self.o[j], j]) < self.tol:
                self.er = -1
                self.add_step({
                    "operation": "Diagonal Check",
                    "description": f"Zero diagonal element at position {j}",
                    "status": "FAILED"
                })
                return

            # Calculate U(j, i) - upper triangular part
            for i in range(j + 1, self.n):
                sum_val = self.a[self.o[j], i]
                for k in range(j):
                    sum_val = super().round_sig_fig(sum_val - self.a[self.o[j], k] * self.a[self.o[k], i])
                self.a[self.o[j], i] = super().round_sig_fig(sum_val / self.a[self.o[j], j])  # U(j, i)

            self.add_step({
                "operation": f"Compute U Row {j}",
                "description": f"Calculate upper triangular elements for row {j}",
                "row": j,
                "matrix_state": self.a.copy().tolist()
            })

    def substitute(self):
        """
        Forward and backward substitution to solve Ly=b and Ux=y.
        """
        a, o, n, b, x = self.a, self.o, self.n, self.b, self.x

        # Forward substitution to compute y from Ly = b
        y = [0] * n
        y[o[0]] = super().round_sig_fig(b[o[0]] / a[o[0], 0])
        for i in range(1, n):
            sum_val = b[o[i]]
            for j in range(i):
                sum_val = super().round_sig_fig(sum_val - a[o[i], j] * y[o[j]])
            y[o[i]] = super().round_sig_fig(sum_val / a[o[i], i])

            self.add_step({
                "operation": f"Forward Substitution: y[{i}]",
                "description": f"Solve Ly=b for position {i}",
                "position": i,
                "value": float(y[o[i]])
            })

        # Backward substitution to compute x from Ux = y
        x[n - 1] = y[o[n - 1]]  # Last variable
        for i in range(n - 2, -1, -1):
            sum_val = y[o[i]]
            for j in range(i + 1, n):
                sum_val = super().round_sig_fig(sum_val - a[o[i], j] * x[j])
            x[i] = sum_val

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

        # Compute scaled value of pivot row
        big = super().round_sig_fig(abs(a[o[k], k]) / s[o[k]])

        # Search for better pivot
        for i in range(k + 1, n):
            dummy = super().round_sig_fig(abs(a[o[i], k] / s[o[i]]))
            if dummy > big:
                big = dummy
                p = i

        # Swap permutation indices
        if p != k:
            dummy = o[p]
            o[p] = o[k]
            o[k] = dummy

            self.add_step({
                "operation": f"Pivot at Column {k}",
                "description": f"Swap rows {o[k]} and {o[p]} for largest scaled element",
                "column": k,
                "from_row": o[p],
                "to_row": o[k]
            })