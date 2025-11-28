from copy import deepcopy
import numpy as np
from methods.AbstractSolver import AbstractSolver
from methods.GaussElimination import GaussElimination
from methods.GaussJordan import GaussJordan
from methods.Doolittle import Doolittle
from methods.Crout import Crout
from methods.IterativeMethod import IterativeMethod
from System.SystemData import SystemData
from methods.Cholesky import Cholesky


class SolverFactory:
    """
    Factory class to instantiate the correct solver based on the method
    specified in the SystemData DTO. This decouples the GUI from specific solvers.
    """
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
    def validate(data: SystemData) :
        A = np.array(deepcopy(data.A))
        b = np.array(deepcopy(data.b))
        n = data.N
        for k in range(n):
            # Find the row with the largest absolute value in column k
            pivot_row = k
            max_val = abs(A[k, k])
            for i in range(k + 1, n):
                if abs(A[i, k]) > max_val:
                    max_val = abs(A[i, k])
                    pivot_row = i

            # If the pivot is almost zero, skip this column
            if max_val < 1e-12:
                continue

            # Swap current row with pivot_row if needed
            if pivot_row != k:
                A[[k, pivot_row]] = A[[pivot_row, k]]
                b[[k, pivot_row]] = b[[pivot_row, k]]

            # Eliminate entries below pivot
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
                A[i, k] = 0.
                b[i] -= factor * b[k]

        for i in range(n):
            if np.all(np.abs(A[i]) < 1e-12) and abs(b[i]) > 1e-12:
                # return {"Success": False, "errorMessage": "System is inconsistent"}
                raise ValueError(f"System is inconsistent (row {i} is all zeros in A but b != 0)")
            if np.all(np.abs(A[i]) < 1e-12) and abs(b[i]) < 1e-12:
                # Row is dependent -> may have infinitely many solutions
                # return {"Success": False,"errorMessage": "System has infinite number of solutions"}
                raise ValueError(f"System has infinite number of solutions")


    @staticmethod
    def get_solver(data: SystemData) -> AbstractSolver:


        print(data.method)

        if data.method == "LU Decomposition":
            data.method = data.params["LU Form"]


        solver_class = SolverFactory.SOLVERS.get(data.method)

        if data.method == "Jacobi-Iteration":
            data.params["Jacobi"] = 1
        else:
            data.params["Jacobi"] = 0

        if data.method != "Jacobi-Iteration" and data.method != "Gauss-Seidel":
            SolverFactory.validate(data)

        if not solver_class:
            raise ValueError(f"Solver for method '{data.method}' is not implemented.")
        # Instantiate the correct solver with the DTO
        return solver_class(data)