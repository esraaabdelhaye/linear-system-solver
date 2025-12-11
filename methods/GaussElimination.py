import numpy as np
from System.SystemData import SystemData
from methods.AbstractSolver import AbstractSolver
from typing import Dict, Any


class GaussElimination(AbstractSolver):
    """
    Solves a system of linear equations using Gaussian Elimination with partial pivoting.
    Supports:
    - Optional scaled partial pivoting (Bonus Feature #3)
    - Step-by-step tracking for single-step mode (Bonus Feature #1)

    Steps:
    1. Forward elimination with partial (scaled) pivoting
    2. Back substitution
    3. Rounding to specified precision at each step
    """

    def __init__(self, data: SystemData):
        """
        Initialize Gauss Elimination solver.

        Args:
            data: SystemData object containing A, b, precision, and parameters
        """
        super().__init__(data)
        self.use_scaling = data.use_scaling

    def solve(self) -> Dict[str, Any]:
        """
        Solve the system using Gaussian Elimination with back substitution.

        Returns:
            Dictionary with success status, solution vector x, and steps (if tracked)

        Raises:
            ValueError: If zero pivot is encountered or system is invalid
        """
        print("Solving: Gauss Elimination")

        A = self.A.astype(float).copy()
        b = self.b.astype(float).copy()

        # Record initial state for step mode
        self.add_step({
            "operation": "Initial System",
            "description": f"System of {self.n} equations",
            "matrix_A": A.copy().tolist(),
            "vector_b": b.copy().tolist()
        })

        # Get scaling factors (used only if use_scaling=True)
        scales = self.get_scales() if self.use_scaling else np.ones(self.n)

        if self.use_scaling:
            self.add_step({
                "operation": "Scaling Factors Computed",
                "description": "Max absolute value per row for scaled pivoting",
                "scales": [float(s) for s in scales]
            })

        # Forward Elimination with Partial Pivoting
        for k in range(self.n - 1):

            # Partial Pivoting (with optional scaling)
            max_ratio = 0
            pivot_row = k

            for i in range(k, self.n):
                ratio = abs(A[i][k]) / scales[i]
                if ratio > max_ratio:
                    max_ratio = ratio
                    pivot_row = i

            # Swap rows if needed
            if pivot_row != k:
                A[[k, pivot_row]] = A[[pivot_row, k]]
                b[[k, pivot_row]] = b[[pivot_row, k]]
                if self.use_scaling:
                    scales[[k, pivot_row]] = scales[[pivot_row, k]]

                self.add_step({
                    "operation": f"Pivot: Swap Rows",
                    "description": f"Swap row {k} with row {pivot_row} (largest scaled element in column {k})",
                    "from_row": k,
                    "to_row": pivot_row,
                    "matrix_A": A.copy().tolist(),
                    "vector_b": b.copy().tolist()
                })

            # Check for zero pivot
            if abs(A[k][k]) < 1e-10:
                raise ValueError(f"Zero pivot encountered at row {k} during elimination.")

            # Elimination step: eliminate entries below pivot
            for i in range(k + 1, self.n):
                if A[k][k] != 0:
                    factor = self.round_sig_fig(A[i][k] / A[k][k])

                    # Record elimination step
                    self.add_step({
                        "operation": f"Eliminate Row {i}",
                        "description": f"R{i} = R{i} - ({factor:.6f}) × R{k}",
                        "pivot_row": k,
                        "elimination_row": i,
                        "factor": float(factor),
                        "matrix_A": A.copy().tolist(),
                        "vector_b": b.copy().tolist()
                    })

                    # Perform elimination
                    A[i, k:] = self.subtract_with_rounding(A[i, k:], factor * A[k, k:])
                    b[i] = self.round_sig_fig(b[i] - factor * b[k])

        # Check final pivot
        if abs(A[self.n - 1][self.n - 1]) < 1e-10:
            raise ValueError("Zero pivot encountered in final row.")

        self.add_step({
            "operation": "Forward Elimination Complete",
            "description": "Upper triangular form achieved",
            "matrix_A": A.copy().tolist(),
            "vector_b": b.copy().tolist()
        })

        # Back Substitution
        x = np.zeros(self.n)

        for i in reversed(range(self.n)):
            sum_ax = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = self.round_sig_fig((b[i] - sum_ax) / A[i][i])
            x[i] = self.round_sig_fig(x[i])

            self.add_step({
                "operation": f"Back Substitution: x[{i}]",
                "description": f"x[{i}] = ({b[i]:.6f} - sum) / {A[i][i]:.6f}",
                "index": i,
                "value": float(x[i]),
                "solution_partial": x.copy().tolist()
            })

        self.add_step({
            "operation": "Solution Found",
            "description": "Back substitution complete",
            "solution": x.copy().tolist()
        })

        print(f"Solution: {x}")
        return {
            "success": True,
            "sol": x,
            "steps": self.steps
        }

    def subtract_with_rounding(self, row, vec, adjust=None):
        """
        Subtract vec from row element-wise with rounding.

        Args:
            row: Row vector
            vec: Vector to subtract
            adjust: Rounding function (uses self.round_sig_fig if not provided)

        Returns:
            Result vector with rounding applied
        """
        if adjust is None:
            adjust = self.round_sig_fig
        return np.array([adjust(a - x) for a, x in zip(row, vec)])