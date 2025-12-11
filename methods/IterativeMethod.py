import numpy as np
from System.SystemData import SystemData
from methods.AbstractSolver import AbstractSolver
from typing import Dict, Any


class IterativeMethod(AbstractSolver):
    """
    Implements iterative techniques (Jacobi and Gauss-Seidel) to solve a system of linear equations A*x = b.
    Supports step-by-step tracking for single-step mode (Bonus Feature #1)

    Constructor: Initializes the solver with system data and iteration parameters.
    """

    def __init__(self, data: SystemData):
        """
        Initialize iterative solver.

        Args:
            data: SystemData object with matrix, initial guess, and stopping criteria
        """
        # Call the constructor of the base class (AbstractSolver)
        super().__init__(data)

        # Instance variables extracted from the SystemData object
        self.X = np.array(data.params["Initial Guess"], dtype=float)  # x^(0)
        self.iterations = data.params["max_iter_var"]  # Max iterations
        self.tol = data.params["error_tol_var"]  # Error tolerance
        self.jacobi = data.params["Jacobi"]  # True for Jacobi, False for Gauss-Seidel

    def solve(self) -> Dict[str, Any]:
        """
        Solves the system A*x = b using either Jacobi or Gauss-Seidel method.

        Returns:
            Dictionary with success status, solution, iterations, and steps
        """
        method_name = "Jacobi" if self.jacobi else "Gauss-Seidel"
        print(f"Solving: {method_name} Iteration")

        n = len(self.b)
        initial_guess = self.X.copy()

        # Record initial state for step mode
        self.add_step({
            "operation": "Initial Guess",
            "description": f"Starting with x^(0)",
            "iteration": 0,
            "x_values": initial_guess.copy().tolist(),
            "error": 0
        })

        # Main iteration loop
        for it in range(self.iterations):
            # Store the vector from current iteration (x^(k)) before computing next one
            old_x = self.X.copy()

            # Loop over each equation/unknown
            for i in range(n):
                """
                Iterative formula derived from: A[i, i] * X[i] = b[i] - sum(A[i, j] * X[j]) for j != i
                X[i]^(k+1) = (1 / A[i, i]) * (b[i] - sum_j!=i (A[i, j] * X[j]))
                """

                if self.jacobi:
                    # --- Jacobi Method Implementation ---
                    # Uses ALL values from the previous iteration (old_x)
                    sum1 = self.dot_with_rounding(self.A[i, :i], old_x[:i], self.round_sig_fig)
                    sum2 = self.dot_with_rounding(self.A[i, i + 1:], old_x[i + 1:], self.round_sig_fig)
                else:
                    # --- Gauss-Seidel Method Implementation ---
                    # Uses NEWLY computed values (self.X) for j < i
                    # and OLD values (old_x) for j > i
                    sum1 = self.dot_with_rounding(self.A[i, :i], self.X[:i], self.round_sig_fig)
                    sum2 = self.dot_with_rounding(self.A[i, i + 1:], old_x[i + 1:], self.round_sig_fig)

                # Compute new value for X[i]
                self.X[i] = self.round_sig_fig((self.b[i] - sum1 - sum2) / self.A[i, i])

            # Calculate convergence error
            error = np.max(np.abs((self.X - old_x) / (self.X + 1e-12)))

            # Record iteration step for step mode
            self.add_step({
                "operation": f"Iteration {it + 1}",
                "description": f"{method_name} iteration update",
                "iteration": it + 1,
                "x_values": self.X.copy().tolist(),
                "previous_x": old_x.copy().tolist(),
                "error": float(error),
                "converged": error < self.tol
            })

            # Check for convergence
            if error < self.tol:
                self.add_step({
                    "operation": "Convergence Achieved",
                    "description": f"Error {error:.6e} < Tolerance {self.tol:.6e}",
                    "final_iteration": it + 1,
                    "final_solution": self.X.copy().tolist(),
                    "final_error": float(error)
                })

                return {
                    "success": True,
                    "sol": self.X,
                    "iterations": it + 1,
                    "error": error,
                    "steps": self.steps
                }

        # Failed to converge
        self.add_step({
            "operation": "Convergence Failed",
            "description": f"Did not converge after {self.iterations} iterations",
            "final_iteration": self.iterations,
            "final_error": error,
            "last_solution": self.X.copy().tolist()
        })

        raise ValueError(f"Couldn't reach tolerance {self.tol} in {self.iterations} iterations. Last error: {error}")

    def dot_with_rounding(self, row, vec, adjust):
        """
        Helper method to calculate dot product with intermediate rounding.
        This simulates limited floating-point precision.

        Args:
            row: Row vector
            vec: Column vector
            adjust: Rounding function

        Returns:
            Dot product result with rounding applied
        """
        total = 0
        # Iterate over the elements simultaneously
        for a, x in zip(row, vec):
            # Multiply elements and apply rounding, then add to total
            total += adjust(a * x)
        # Apply rounding to the final sum
        return adjust(total)