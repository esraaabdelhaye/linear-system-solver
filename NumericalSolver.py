import tkinter as tk
from tkinter import messagebox
import time
from typing import List, Tuple, Dict, Any, Optional
import System.SystemData as SystemData
import methods.SolverFactory as SolverFactory



class NumericalSolver:
    """
    The coordinator class: handles input parsing, DTO creation, and dispatching
    the solving task to the appropriate solver via the Factory.
    """

    def parse_input(self, entry_widgets: List[List[tk.Entry]], N: int) -> Optional[
        Tuple[List[List[float]], List[float]]]:
        """
        Reads numerical data from the Tkinter Entry grid into the augmented matrix [A|b].
        (Specification 1b: Bullet-proof input validation)
        """
        A = []
        b = []

        if N == 0:
            messagebox.showerror("Input Error", "Number of variables (N) must be > 0.")
            return None

        try:
            for i in range(N):
                row_a = []
                # Iterate through columns for coefficients A[i][j]
                for j in range(N):
                    value = entry_widgets[i][j].get().strip()
                    # Convert to float. Empty input is treated as 0.0 (Specification 1d)
                    row_a.append(float(value) if value else 0.0)

                    # Constant b[i] (last column)
                value_b = entry_widgets[i][N].get().strip()
                row_b = float(value_b) if value_b else 0.0

                A.append(row_a)
                b.append(row_b)

        except ValueError:
            messagebox.showerror("Input Error",
                                 "All coefficients and constants must be valid numbers (or left blank for 0).")
            return None

        # The grid structure guarantees N variables = N equations (Specification 1c)
        return A, b

    def solve(self, data: SystemData) -> Dict[str, Any]:
        """
        Dispatches the solving request to the correct solver implementation.
        """
        start_time = time.time()

        try:
            # Factory provides the specific solver instance
            solver = SolverFactory.get_solver(data)
            results = solver.solve()

            # Add metadata back to results for GUI display
            results["method_used"] = data.method
            results["precision"] = data.precision
            return results

        except ValueError as e:
            # Handle solver-specific errors (e.g., convergence failure, singularity)
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

