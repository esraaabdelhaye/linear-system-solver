import numpy as np
import math
import time
from AbstractSolver import AbstractSolver

class CholeskySolver(AbstractSolver):
    """
    Implements Cholesky decomposition for solving symmetric, positive-definite systems.
    Supports optional row scaling and logs all steps.
    """

    def __init__(self, A, b, precision=6, single_step=False, use_scaling=False):
        super().__init__(A, b, precision, single_step)
        self.use_scaling = use_scaling
        # Compute scales if scaling is used, otherwise use ones
        self.scales = self.get_scales() if use_scaling else [1]*self.n

    def get_scales(self):
        """
        Compute scaling factors: maximum absolute value per row.
        Avoid division by zero by using 1 if row is all zeros.
        """
        return [max(abs(val) for val in row) or 1 for row in self.A]

    def solve(self):
        """
        Main method: performs Cholesky decomposition, forward/backward substitution, and logs steps.
        Returns a dictionary containing solution, L matrix, execution time, and steps.
        """
        self.validate()
        start_time = time.time()  # Start timing
        A = np.copy(self.A).astype(float)
        b = np.copy(self.b).astype(float)

        
        # Optional Scaling
        if self.use_scaling:
            for i in range(self.n):
                A[i] /= self.scales[i]
                b[i] /= self.scales[i]
            # Convert scales to float for clean output
            scales_float = [float(s) for s in self.scales]
            self.add_step("Scaling Applied", {
                "scaled_A": A.tolist(),
                "scaled_b": b.tolist(),
                "scales": scales_float
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






# A = [
#     [2, 1, 1, 1, 1],
#     [1, 2, 1, 1, 1],
#     [1, 1, 2, 1, 1],
#     [1, 1, 1, 2, 1],
#     [1, 1, 1, 1, 2]
# ]
# b = [1, 2, 3, 4, 5]

# solver = CholeskySolver(A, b, precision=6, single_step=True, use_scaling=True)
# result = solver.solve()
# print("Solution:", result["solution"])

# print("\n=== Steps ===")
# for i, step in enumerate(result["steps"], start=1):
#     print(f"\n--- Step {i}: {step['label']} ---")
#     print(step["value"])
