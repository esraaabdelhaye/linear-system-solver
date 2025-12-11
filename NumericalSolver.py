import tkinter as tk
from tkinter import messagebox
import time
from typing import List, Tuple, Dict, Any, Optional
from System.SystemData import SystemData
from methods.SolverFactory import SolverFactory
from methods.AbstractSolver import AbstractSolver


class NumericalSolver:
    """
    Coordinator class responsible for:
      - Parsing user input from the GUI
      - Creating the SystemData DTO
      - Dispatching the solve request to the correct solver via the Factory

    This keeps GUI code clean and maintains separation of concerns.
    """

    def parse_input(self, entry_widgets: List[List[tk.Entry]], N: int) -> Optional[
        Tuple[List[List[float]], List[float]]]:
        """
        Reads numerical data from the Tkinter Entry grid into the augmented matrix [A|b].
        Returns:
            (A, b) if input is valid, otherwise None.

        Note:
        - Empty cells are treated as 0.0
        """
        A = []
        b = []

        # Basic validation: N must be positive
        if N == 0:
            messagebox.showerror("Input Error", "Number of variables (N) must be > 0.")
            return None

        try:
            for i in range(N):
                row_a = []
                # Iterate through columns for coefficients A[i][j]
                for j in range(N):
                    value = entry_widgets[i][j].get().strip()
                    # Convert to float. Empty input is treated as 0.0
                    row_a.append(float(value) if value else 0.0)

                    # Constant b[i] (last column)
                value_b = entry_widgets[i][N].get().strip()
                row_b = float(value_b) if value_b else 0.0

                A.append(row_a)
                b.append(row_b)

        # When conversion to float fails
        except ValueError:
            messagebox.showerror("Input Error",
                                 "All coefficients and constants must be valid numbers (or left blank for 0).")
            return None

        # The grid structure guarantees N variables = N equations
        return A, b

    def solve(self, data: SystemData) -> Dict[str, Any]:
        # print("solve from Numberical Solver")
        """
        Dispatches the solving request to the correct solver implementation.
        """
        #Start timing
        start_time = time.time()

        try:

            # Factory provides the specific solver instance
            solver = SolverFactory.get_solver(data)
            # Perform the computation
            results = solver.solve()

            # print("results: ", results)

            # Add metadata back to results for GUI display
            results["method_used"] = data.method
            results["precision"] = data.precision
            results["execution_time"] = time.time() - start_time

            return results

        except ValueError as e:
            # Handle solver-specific errors (for example: convergence failure, singularity)
            return {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            # Catch unexpected Python errors
            return {
                "success": False,
                "error_message": f"An unexpected error occurred: {str(e)}",
                "execution_time": time.time() - start_time,
            }

