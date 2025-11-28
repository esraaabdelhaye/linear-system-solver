
from System.SystemData import SystemData
from methods.IterativeMethod import IterativeMethod
from methods.Crout import Crout
from methods.Dolittle import Doolittle
from methods.AbstractSolver import AbstractSolver
from methods.GaussElimination import GaussElimination
from methods.GaussJordan import GaussJordan


class SolverFactory:
    """
    Factory class to instantiate the correct solver based on the method
    specified in the SystemData DTO. This decouples the GUI from specific solvers.
    """
    SOLVERS = {
        "Gauss Elimination": GaussElimination,
        "Gauss-Jordan": GaussJordan,
        "Doolittle Form": Doolittle,
        "Crout Form": Crout,
        # "Cholesky Form": CholeskySolver
        "iterative-method": IterativeMethod,
    }

    @staticmethod
    def get_solver(data: SystemData) -> AbstractSolver:
        """Returns an instance of the specific solver class."""
        method = SolverFactory.SOLVERS.get(data.method)
        if method == "LU Decomposition":
            method = data.params["LU Form"]
        solver_class = SolverFactory.SOLVERS.get(method)
        if not solver_class:
            raise ValueError(f"Solver for method '{data.method}' is not implemented.")
        # Instantiate the correct solver with the DTO
        return solver_class(data)