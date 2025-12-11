import numpy as np
from System.SystemData import SystemData
from methods.AbstractSolver import AbstractSolver
from typing import Dict, Any


class GaussJordan(AbstractSolver):
    """
    Solves a system of linear equations using Gauss-Jordan elimination.
    Reduces the augmented matrix to reduced row echelon form (RREF).
    Supports:
    - Optional scaled partial pivoting (Bonus Feature #3)
    - Step-by-step tracking for single-step mode (Bonus Feature #1)

    Steps:
    1. For each column k:
       - Find pivot using (optional) scaled partial pivoting
       - Normalize pivot row to make diagonal = 1
       - Eliminate ALL other rows (above and below)
    2. Solution is directly the RHS vector after RREF transformation
    """

    def __init__(self, data: SystemData):
        """
        Initialize Gauss-Jordan solver.

        Args:
            data: SystemData object containing A, b, precision, and parameters
        """
        super().__init__(data)
        self.use_scaling = data.use_scaling

    def solve(self) -> Dict[str, Any]:
        """
        Solve the system using Gauss-Jordan elimination to RREF.

        Returns:
            Dictionary with success status, solution vector x, and steps (if tracked)

        Raises:
            ValueError: If zero pivot is encountered
        """
        print("Solving: Gauss-Jordan Elimination")

        A = self.A.astype(float).copy()
        b = self.b.astype(float).copy()

        # Record initial state for step mode
        self.add_step({
            "operation": "Initial System",
            "description": f"System of {self.n} equations",
            "matrix_A": A.copy().tolist(),
            "vector_b": b.copy().tolist()
        })

        # Get scaling factors for partial pivoting
        scales = self.get_scales() if self.use_scaling else np.ones(self.n)

        if self.use_scaling:
            self.add_step({
                "operation": "Scaling Factors Computed",
                "description": "Max absolute value per row for scaled pivoting",
                "scales": [float(s) for s in scales]
            })

        # Gauss-Jordan Elimination (to RREF)
        for k in range(self.n):

            # Partial Pivoting with optional scaling
            max_ratio = 0
            pivot_row = k

            for i in range(k, self.n):
                ratio = abs(A[i][k]) / scales[i] if self.use_scaling else abs(A[i][k])
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
            pivot = A[k][k]
            if abs(pivot) < 1e-10:
                raise ValueError(f"Zero pivot encountered at row {k}.")

            # Normalize pivot row (make diagonal element = 1)
            A[k] = self.round_sig_fig(A[k] / pivot)
            b[k] = self.round_sig_fig(b[k] / pivot)

            self.add_step({
                "operation": f"Normalize Row {k}",
                "description": f"Divide row {k} by {pivot:.6f} (make diagonal = 1)",
                "pivot_row": k,
                "pivot_value": float(pivot),
                "matrix_A": A.copy().tolist(),
                "vector_b": b.copy().tolist()
            })

            # Eliminate ALL other rows (above and below)
            for i in range(self.n):
                if i == k:
                    continue

                factor = A[i][k]
                if abs(factor) > 1e-10:
                    self.add_step({
                        "operation": f"Eliminate Row {i}",
                        "description": f"R{i} = R{i} - ({factor:.6f}) × R{k}",
                        "pivot_row": k,
                        "elimination_row": i,
                        "factor": float(factor),
                        "matrix_A": A.copy().tolist(),
                        "vector_b": b.copy().tolist()
                    })

                    A[i] = self.subtract_with_rounding(A[i], factor * A[k])
                    b[i] = self.round_sig_fig(b[i] - factor * b[k])

        self.add_step({
            "operation": "Reduced Row Echelon Form (RREF)",
            "description": "Matrix reduced to identity form",
            "matrix_A": A.copy().tolist(),
            "vector_b": b.copy().tolist()
        })

        # Solution is directly in b vector (since A is now identity matrix)
        x = b.copy()

        # Apply precision rounding to final solution
        for i in range(self.n):
            x[i] = self.round_sig_fig(x[i])

        self.add_step({
            "operation": "Solution Found",
            "description": "RREF complete - solution is RHS vector",
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