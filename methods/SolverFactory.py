
from methods.AbstractSolver import AbstractSolver
from methods.GaussElimination import GaussElimination
from methods.GaussJordan import GaussJordan
from methods.Doolittle import Doolittle
from methods.Crout import Crout
from methods.IterativeMethod import IterativeMethod
from System.SystemData import SystemData


class SolverFactory:
    """
    Factory class to instantiate the correct solver based on the method
    specified in the SystemData DTO. This decouples the GUI from specific solvers.
    """
    SOLVERS = {
        "Gauss Elimination": GaussElimination,
        "Gauss-Jordan": GaussJordan,
        "doolittle": Doolittle,
        "crout": Crout,
        # "cholesky": CholeskySolver
        "Jacobi-Iteration": IterativeMethod,
        "Gauss-Seidel": IterativeMethod,
    }

    @staticmethod
    def get_solver(data: SystemData) -> AbstractSolver:
        """Returns an instance of the specific solver class."""
        method = SolverFactory.SOLVERS.get(data.method)
        if method == "LU Decomposition":
            method = data.params["LU Form"]

        solver_class = SolverFactory.SOLVERS.get(data.method)

        if data.method == "Jacobi-Iteration":
            data.params["Jacobi"] = 1
        else:
            data.params["Jacobi"] = 0



        if not solver_class:
            raise ValueError(f"Solver for method '{data.method}' is not implemented.")
        # Instantiate the correct solver with the DTO
        return solver_class(data)