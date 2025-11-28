import timeit
import numpy as np
from System.SystemData import SystemData
from methods.AbstractSolver import AbstractSolver
from typing import Dict, Any


# The IterativeMethod class inherits from AbstractSolver
# iterative techniques (Jacobi and Gauss-Seidel) to solve a system of linear equations A*x = b.
class IterativeMethod(AbstractSolver):
    # Constructor: Initializes the solver with system data and iteration parameters.
    def __init__(self, data: SystemData):
        # Call the constructor of the base class (AbstractSolver).
        super().__init__(data)

        # Instance variables (params) are extracted from the SystemData object.
        # Initial guess vector (x^(0)).
        self.X = np.array(data.params["Initial Guess"], dtype=float)
        # Maximum number of iterations allowed.
        self.iterations = data.params["max_iter_var"]
        # Error tolerance (stopping criterion).
        self.tol = data.params["error_tol_var"]
        # Boolean flag: True for Jacobi method, False for Gauss-Seidel method.
        self.jacobi = data.params["Jacobi"]
        # Note: self.A, self.b, and self.round_sig_fig are inherited from AbstractSolver.

    # Solves the system of linear equations A*x = b using either Jacobi or Gauss-Seidel.
    def solve(self) -> Dict[str, Any]:
        # Number of equations/unknowns.
        n = len(self.b)

        # Main iteration loop.
        for it in range(self.iterations):
            # Store the vector from the current iteration (x^(k)) before computing the next one.
            old_x = self.X.copy()

            # Loop over each equation/unknown (i = 0 to n-1).
            for i in range(n):
                # The iterative formula is derived from: A[i, i] * X[i] = b[i] - sum(A[i, j] * X[j]) for j != i
                # X[i]^(k+1) = (1 / A[i, i]) * (b[i] - sum_j!=i (A[i, j] * X[j]))

                if self.jacobi:
                    # --- Jacobi Method Implementation ---
                    # Jacobi uses ALL values from the previous iteration (old_x) to compute the new X[i].

                    # sum1: sum of A[i, j] * old_x[j] for j < i (lower triangle part).
                    sum1 = self.dot_with_rounding(self.A[i, :i], old_x[:i], self.round_sig_fig)
                    # sum2: sum of A[i, j] * old_x[j] for j > i (upper triangle part).
                    sum2 = self.dot_with_rounding(self.A[i, i + 1:], old_x[i + 1:], self.round_sig_fig)
                else:
                    # --- Gauss-Seidel Method Implementation ---
                    # Gauss-Seidel uses the NEWLY computed values (self.X) for j < i
                    # and the OLD values (old_x) for j > i.

                    # sum1: sum of A[i, j] * self.X[j] for j < i.
                    # These x values (self.X[:i]) are the NEW updated ones
                    # because we’ve already computed them earlier in this iteration (for indices < i).
                    sum1 = self.dot_with_rounding(self.A[i, :i], self.X[:i], self.round_sig_fig)

                    # sum2: sum of A[i, j] * old_x[j] for j > i.
                    # These x values (old_x[i + 1:]) are the OLD ones
                    # because we haven’t computed their new value in this iteration yet.
                    sum2 = self.dot_with_rounding(self.A[i, i + 1:], old_x[i + 1:], self.round_sig_fig)

                # Compute the new value for X[i] using the equation derived from A*x=b.
                # X[i] = (b[i] - sum_j!=i (A[i, j] * X[j])) / A[i, i]
                self.X[i] = self.round_sig_fig((self.b[i] - sum1 - sum2) / self.A[i, i])

            # --- Convergence Check ---
            # Calculate the maximum absolute relative error between the new (self.X) and old (old_x) vectors.
            # error = max(| (X[i]^(k+1) - X[i]^(k)) / X[i]^(k+1) |)
            # A small epsilon (1e-12) is added to the denominator to protect against division by zero
            # (though mathematically X[i]^(k+1) should not be zero if the system is well-behaved).
            error = np.max(np.abs((self.X - old_x) / (self.X + 1e-12)))

            # Check if the calculated error is less than the user-defined tolerance.
            if error < self.tol:
                # If converged, return the successful result, the solution vector, and the number of iterations.
                return {"success": True, "sol": self.X,
                        "iterations": it + 1}  # it is 0-indexed, so we add 1 for the count.

        # If the loop finishes without meeting the tolerance, raise an error.
        raise ValueError("Couldn't reach that tolerance in the given number of iterations")

    # Helper method to calculate the dot product of a row and a vector with intermediate rounding.
    # This is used to simulate limited floating-point precision.
    def dot_with_rounding(self, row, vec, adjust):
        total = 0
        # Iterate over the elements of the row and vector simultaneously.
        for a, x in zip(row, vec):
            # Multiply the elements and apply the rounding function (adjust), then add to total.
            total += adjust(a * x)
        # Apply the rounding function to the final sum before returning.
        return adjust(total)