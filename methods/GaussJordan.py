import numpy as np
from System.SystemData import SystemData
from methods.AbstractSolver import AbstractSolver
from typing import  Dict, Any

class GaussJordan(AbstractSolver):
    """
    Solves a system of linear equations using Gauss-Jordan elimination.
    Reduces the augmented matrix to reduced row echelon form (RREF).
    """

    def __init__(self, data: SystemData):
        """
        Initialize Gauss-Jordan solver.

        Args:
            A: Coefficient matrix
            b: Constants vector
            precision: Number of significant figures
            single_step: Enable step-by-step recording
            use_scaling: Enable scaled partial pivoting (Bonus #3)
        """
        super().__init__(data)
        self.use_scaling = data.params["use_scaling"]

    def solve(self) -> Dict[str, Any]:
        """
        Solve the system using Gauss-Jordan elimination to RREF.

        Returns:
            Solution vector x

        Raises:
            ValueError: If zero pivot is encountered
        """
        self.validate()

        A = self.A.astype(float).copy()
        b = self.b.astype(float).copy()

        # if self.single_step:
        #     self.add_step(("Initial System", A.copy(), b.copy()))

        # Get scaling factors
        # scales = self.get_scales() if self.use_scaling else np.ones(self.n)

        # Gauss-Jordan Elimination (to RREF)
        for k in range(self.n):

            # Partial Pivoting with optional scaling
            max_ratio = 0
            pivot_row = k

            for i in range(k, self.n):
                ratio = abs(A[i][k]) /1
                #scales[i]
                if ratio > max_ratio:
                    max_ratio = ratio
                    pivot_row = i

            # Swap rows if needed
            if pivot_row != k:
                A[[k, pivot_row]] = A[[pivot_row, k]]
                b[[k, pivot_row]] = b[[pivot_row, k]]
                # if self.use_scaling:
                #     scales[[k, pivot_row]] = scales[[pivot_row, k]]

                # if self.single_step:
                #     self.add_step((f"Pivot: Swap row {k} ↔ row {pivot_row}", A.copy(), b.copy()))

            # Check for zero pivot (basic validation)
            pivot = A[k][k]
            if abs(pivot) < 1e-10:
                raise ValueError("Zero pivot encountered during elimination.")

            # Normalize pivot row (make diagonal element = 1)
            A[k] /= pivot
            b[k] /= pivot

            # if self.single_step:
            #     self.add_step((f"Normalize: R{k} = R{k} / {pivot:.4f}", A.copy(), b.copy()))

            # Eliminate ALL other rows (above and below)
            for i in range(self.n):
                if i == k:
                    continue

                factor = A[i][k]
                if abs(factor) > 1e-10:  # Only eliminate if factor is significant
                    A[i] -= factor * A[k]
                    b[i] -= factor * b[k]

        #             if self.single_step:
        #                 self.add_step((f"Eliminate: R{i} = R{i} - ({factor:.4f}) × R{k}",
        #                                A.copy(), b.copy()))
        #
        # if self.single_step:
        #     self.add_step(("Reduced Row Echelon Form (RREF)", A.copy(), b.copy()))

        # Solution is directly in b vector (since A is now identity matrix)
        x = b.copy()

        # Apply precision
        for i in range(self.n):
            x[i] = self.round_sig_fig(x[i])

        return {"sol": x}