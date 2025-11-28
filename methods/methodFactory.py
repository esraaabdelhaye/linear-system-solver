import System.SystemData as SystemData
import methods.AbstractSolver as AbstractSolver
import methods.GaussElimination as GaussEliminationSolver
import methods.iterative_method as IterativeMethodSolver
import methods.GaussJordan as GaussJordanSolver

class SolverFactory:
    """
    Factory class to instantiate the correct solver based on the method
    specified in the SystemData DTO. This decouples the GUI from specific solvers.
    """
    SOLVERS = {
        "Gauss Elimination": GaussEliminationSolver,
        "Gauss-Jordan": GaussEliminationSolver,  # Placeholder, should be GaussJordanSolver
        "LU Decomposition": GaussEliminationSolver,  # Placeholder, should be LUSolver
        "iterative-method": IterativeMethodSolver,
    }

    @staticmethod
    def get_solver(data: SystemData) -> AbstractSolver:
        """Returns an instance of the specific solver class."""
        solver_class = SolverFactory.SOLVERS.get(data.method)
        if not solver_class:
            raise ValueError(f"Solver for method '{data.method}' is not implemented.")
        # Instantiate the correct solver with the DTO
        return solver_class(data)