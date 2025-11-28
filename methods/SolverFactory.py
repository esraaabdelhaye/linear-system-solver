from copy import deepcopy
import numpy as np

# Import abstract solver class and all specific solver implementations
from methods.AbstractSolver import AbstractSolver
from methods.GaussElimination import GaussElimination
from methods.GaussJordan import GaussJordan
from methods.Doolittle import Doolittle
from methods.Crout import Crout
from methods.IterativeMethod import IterativeMethod
from methods.Cholesky import Cholesky

# Import the data transfer object (DTO) that holds the system data
from System.SystemData import SystemData


class SolverFactory:
    """
    Factory class to instantiate the correct solver based on the method
    specified in the SystemData DTO. This decouples the GUI (or other clients)
    from needing to know about specific solver implementations.
    """

    # Dictionary mapping method names to solver classes
    SOLVERS = {
        "Gauss Elimination": GaussElimination,
        "Gauss-Jordan": GaussJordan,
        "Doolittle": Doolittle,
        "Crout": Crout,
        "Cholesky": Cholesky,
        "Jacobi-Iteration": IterativeMethod,
        "Gauss-Seidel": IterativeMethod,
    }

    @staticmethod
    def validate(data: SystemData):
        """
        Validates the given system of equations (A * x = b) for:
        - Inconsistency (no solution)
        - Dependency (infinite solutions)
        This method performs a partial Gaussian elimination with partial pivoting
        to detect problematic rows.

        Raises:
            ValueError: if the system is inconsistent or has infinite solutions.
        """

        # Make deep copies of A and b to avoid modifying the original data
        A = np.array(deepcopy(data.A))
        b = np.array(deepcopy(data.b))
        n = data.N  # Number of equations / size of the system

        # Gaussian elimination with partial pivoting
        for k in range(n):
            pivot_row = k
            max_val = abs(A[k, k])

            # Find the pivot row (row with largest absolute value in column k)
            for i in range(k + 1, n):
                if abs(A[i, k]) > max_val:
                    max_val = abs(A[i, k])
                    pivot_row = i

            # Skip column if pivot is extremely small (effectively zero)
            if max_val < 1e-12:
                continue

            # Swap current row with pivot row if needed
            if pivot_row != k:
                A[[k, pivot_row]] = A[[pivot_row, k]]  # Swap rows in A
                b[[k, pivot_row]] = b[[pivot_row, k]]  # Swap corresponding entries in b

            # Eliminate entries below the pivot in column k
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]  # Subtract multiples of pivot row
                A[i, k] = 0.                   # Explicitly set eliminated element to 0
                b[i] -= factor * b[k]

        # Check for inconsistent or dependent rows
        for i in range(n):
            if np.all(np.abs(A[i]) < 1e-12) and abs(b[i]) > 1e-12:
                # Row has all zeros in A but b != 0 → inconsistent system
                raise ValueError(f"System is inconsistent (row {i} is all zeros in A but b != 0)")
            if np.all(np.abs(A[i]) < 1e-12) and abs(b[i]) < 1e-12:
                # Row has all zeros in A and b → dependent row, infinite solutions
                raise ValueError(f"System has infinite number of solutions")

    @staticmethod
    def get_solver(data: SystemData) -> AbstractSolver:
        """
        Factory method to return an instance of the correct solver class
        based on the 'method' specified in the SystemData object.

        Args:
            data (SystemData): DTO containing system data and solver parameters.

        Returns:
            AbstractSolver: An instance of the solver class corresponding to the method.

        Raises:
            ValueError: if the method is not implemented or system validation fails.
        """

        print(data.method)  # Debug: print the selected method

        # Handle LU decomposition special case
        if data.method == "LU Decomposition":
            # Choose the form of LU decomposition (Doolittle, Crout, or Cholesky) from params
            data.method = data.params["LU Form"]

        # Get the solver class corresponding to the method
        solver_class = SolverFactory.SOLVERS.get(data.method)

        # Set a flag for IterativeMethod to distinguish between Jacobi and Gauss-Seidel
        if data.method == "Jacobi-Iteration":
            data.params["Jacobi"] = 1
        else:
            data.params["Jacobi"] = 0

        # Validate the system for direct methods (skip for iterative methods)
        if data.method != "Jacobi-Iteration" and data.method != "Gauss-Seidel":
            SolverFactory.validate(data)

        # If the method is not implemented in the dictionary, raise an error
        if not solver_class:
            raise ValueError(f"Solver for method '{data.method}' is not implemented.")

        # Instantiate and return the solver with the given data
        return solver_class(data)
