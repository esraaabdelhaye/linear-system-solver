
from methods.AbstractSolver import AbstractSolver
from methods.GaussElimination import GaussElimination
from methods.GaussJordan import GaussJordan
from methods.dolittle import Doolittle
from methods.crout import Crout
from methods.iterative_method import iterative_method
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
        "iterative-method": iterative_method,
    }

    @staticmethod
    def get_solver(data: SystemData) -> AbstractSolver:
        """Returns an instance of the specific solver class."""
        solver_class = SolverFactory.SOLVERS.get(data.method)
        if not solver_class:
            raise ValueError(f"Solver for method '{data.method}' is not implemented.")
        # Instantiate the correct solver with the DTO
        return solver_class(data)