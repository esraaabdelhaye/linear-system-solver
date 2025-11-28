import numpy as np
from System.SystemData import SystemData
from methods.AbstractSolver import AbstractSolver
from typing import  Dict, Any

class GaussElimination(AbstractSolver):
    """
    Solves a system of linear equations using Gaussian Elimination with partial pivoting.
    Supports optional scaling and single-step mode.
    """

    def __init__(self, data: SystemData):
        """
        Initialize Gauss Elimination solver.

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
        print("solve: Gauss Elimination")
        """
        Solve the system using Gaussian Elimination with back substitution.

        Returns:
            Solution vector x

        Raises:
            ValueError: If zero pivot is encountered
        """
        # self.validate()

        A = self.A.astype(float).copy()
        b = self.b.astype(float).copy()

        # if self.single_step:
        #     self.add_step(("Initial System", A.copy(), b.copy()))

        # Get scaling factors (used only if use_scaling=True)
        scales = self.get_scales() if self.use_scaling else np.ones(self.n)

        # if self.single_step and self.use_scaling:
        #     self.add_step(("Scaling factors", scales.copy(), None))

        # Forward Elimination with Partial Pivoting
        for k in range(self.n - 1):

            # Partial Pivoting (with optional scaling)
            # Find the row with largest scaled pivot element
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

                if self.single_step:
                    self.add_step((f"Pivot: Swap row {k} ↔ row {pivot_row}", A.copy(), b.copy()))

            # Check for zero pivot (basic validation)
            if abs(A[k][k]) < 1e-10:
                raise ValueError("Zero pivot encountered during elimination.")

            # Elimination step
            for i in range(k + 1, self.n):
                if A[k][k] != 0:  # Extra safety check
                    factor = A[i][k] / A[k][k]
                    A[i, k:] -= factor * A[k, k:]  # Vectorized operation
                    b[i] -= factor * b[k]

                    # if self.single_step:
                    #     self.add_step((f"Eliminate: R{i} = R{i} - ({factor:.4f}) × R{k}",
                    #                    A.copy(), b.copy()))

        # Check final pivot
        if abs(A[self.n - 1][self.n - 1]) < 1e-10:
            raise ValueError("Zero pivot encountered in final row.")

        # if self.single_step:
        #     self.add_step(("Upper Triangular Form", A.copy(), b.copy()))

        # Back Substitution
        x = np.zeros(self.n)

        for i in reversed(range(self.n)):
            sum_ax = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - sum_ax) / A[i][i]
            x[i] = self.round_sig_fig(x[i])

            # if self.single_step:
            #     self.add_step((f"Back-sub: x[{i}] = {x[i]}", None, x.copy()))

        print(x)
        return {"sol": x}