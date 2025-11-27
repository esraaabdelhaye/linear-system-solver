import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
import time
import json
from typing import List, Tuple, Dict, Any, Optional


# --- 1. CORE LOGIC PLACEHOLDERS (To be implemented in detail by you) ---

class NumericalSolver:
    """
    A class to encapsulate all numerical methods for solving systems of linear equations.
    This structure promotes good OOP practices as required by the project.
    """

    def __init__(self, precision: int = 5):
        """Initializes the solver with a default precision."""
        self.precision = precision
        # You will add more attributes and methods here later

    def set_precision(self, precision: int):
        """Sets the number of significant figures for calculations."""
        self.precision = precision

    def parse_input(self, input_text: str) -> Optional[Tuple[List[List[float]], List[float]]]:
        """
        Parses the raw text input (e.g., matrix format) into the Augmented Matrix [A|b].

        Assumed Input Format (Example for 3x3):
        1 2 3 | 10
        4 5 6 | 20
        7 8 9 | 30

        Note: You will need to implement robust, bullet-proof validation here
              to ensure coefficients are numbers and the system is square (N variables = N equations).
        """
        lines = input_text.strip().split('\n')
        N = len(lines)
        A = []
        b = []

        if N == 0:
            messagebox.showerror("Input Error", "The system of equations cannot be empty.")
            return None

        try:
            for i, line in enumerate(lines):
                # Regex to split on spaces, ignoring empty strings, and handling the '|' separator
                parts = [p.strip() for p in re.split(r'\s*\|\s*|\s+', line) if p.strip()]

                # Check for N+1 components (N coefficients + 1 constant)
                if len(parts) != N + 1:
                    messagebox.showerror("Input Error",
                                         f"Row {i + 1} has {len(parts) - 1} coefficients. Expected {N} coefficients for an {N}x{N} system.")
                    return None

                row_a = [float(p) for p in parts[:-1]]
                row_b = float(parts[-1])

                A.append(row_a)
                b.append(row_b)

        except ValueError:
            messagebox.showerror("Input Error", "All coefficients and constants must be valid numbers.")
            return None

        return A, b

    def solve(self, method: str, A: List[List[float]], b: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        The main solving dispatch method. Replace this placeholder with actual
        numerical solving implementations.
        """
        N = len(A)
        start_time = time.time()

        # --- Placeholder Results ---
        solution = [float(i + 1) for i in range(N)]  # Example solution
        iterations = "N/A"
        execution_time = 0.0

        try:
            if method in ["Gauss Elimination", "Gauss-Jordan"]:
                # Implement partial pivoting here (Specification 8)
                time.sleep(0.5)  # Simulate calculation time
                # Your full Gauss Elimination/Gauss-Jordan logic goes here

                # Check for no solution / infinite solutions (Specification 6)
                # if check_for_singularity(A): raise ValueError("Singular Matrix: No unique solution.")

            elif method == "LU Decomposition":
                # Check params for 'LU Form' (Doolittle, Crout, Cholesky)
                lu_form = params.get("LU Form", "Doolittle Form")
                # Implement partial pivoting (Doolittle form only) and LU decomposition logic
                time.sleep(0.7)

            elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
                # Check params for 'Initial Guess' and 'Stopping Condition'
                initial_guess = params.get("Initial Guess", [0.0] * N)
                stop_condition = params.get("Stopping Condition Type", "Number of Iterations")
                stop_value = params.get("Stopping Value", 50)

                # Implement convergence check (diagonal dominance) and iterative logic
                time.sleep(1.0)
                iterations = 35  # Example iteration count

            else:
                raise ValueError("Selected method is not implemented.")

            execution_time = time.time() - start_time

            return {
                "success": True,
                "solution": solution,
                "execution_time": execution_time,
                "iterations": iterations,
                "method_used": method,
                "precision": self.precision
            }

        except ValueError as e:
            return {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            return {
                "success": False,
                "error_message": f"An unexpected error occurred: {str(e)}",
                "execution_time": time.time() - start_time,
            }


# --- 2. GUI APPLICATION CLASS ---

class NumericalSolverGUI:
    """
    The main Tkinter application window.
    """

    def __init__(self, master):
        self.master = master
        master.title("Numerical Linear System Solver (Project Phase 1)")

        # Initialize Solver Backend
        self.solver = NumericalSolver()

        # --- Variables ---
        self.method_var = tk.StringVar(master, value="Gauss Elimination")
        self.precision_var = tk.StringVar(master, value="5")  # Default precision (Spec 4)

        # Dynamic parameter variables
        self.lu_form_var = tk.StringVar(master, value="Doolittle Form")
        self.initial_guess_var = tk.StringVar(master, value="0, 0, 0")  # Example for 3x3
        self.stop_condition_type_var = tk.StringVar(master, value="Number of Iterations")
        self.stop_value_var = tk.StringVar(master, value="50")

        # --- Setup Styles ---
        self.setup_styles()

        # --- Setup Main Frames ---
        self.main_frame = ttk.Frame(master, padding="15 15 15 15")
        self.main_frame.pack(fill='both', expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        # --- Input Frame (Left Side) ---
        self.input_frame = ttk.LabelFrame(self.main_frame, text="1. System Input & Method Selection", padding="10")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.input_frame.columnconfigure(0, weight=1)

        # 1. System of Equations Input (Specification 1)
        ttk.Label(self.input_frame, text="Enter Augmented Matrix [A|b] (e.g., '1 2 | 5' per line):").pack(fill='x',
                                                                                                          pady=(0, 5))
        self.input_text = scrolledtext.ScrolledText(self.input_frame, wrap=tk.WORD, height=8, width=40,
                                                    font=("Consolas", 10))
        self.input_text.pack(fill='both', expand=True, pady=(0, 10))
        # Initial placeholder data for quick testing (3x3 system)
        self.input_text.insert(tk.END, "4 1 -1 | 3\n2 7 1 | 19\n1 -3 12 | 31")

        # 2. Method Selection (Specification 2)
        ttk.Label(self.input_frame, text="2. Choose Solving Method:").pack(fill='x', pady=(5, 5))
        self.method_options = [
            "Gauss Elimination",
            "Gauss-Jordan",
            "LU Decomposition",
            "Jacobi-Iteration",
            "Gauss-Seidel"
        ]
        self.method_dropdown = ttk.Combobox(self.input_frame,
                                            textvariable=self.method_var,
                                            values=self.method_options,
                                            state="readonly")
        self.method_dropdown.pack(fill='x', pady=(0, 10))
        self.method_var.trace_add("write", self.update_parameters_frame)  # Dynamic update

        # 3. Dynamic Parameters Frame (Specification 3)
        self.params_frame = ttk.LabelFrame(self.input_frame, text="3. Method Parameters", padding="10")
        self.params_frame.pack(fill='x', pady=(5, 10))
        self.update_parameters_frame()  # Initial call

        # 4. Precision Input (Specification 4)
        precision_frame = ttk.Frame(self.input_frame)
        precision_frame.pack(fill='x', pady=(5, 5))
        ttk.Label(precision_frame, text="4. Precision (Significant Figures):").pack(side=tk.LEFT)
        self.precision_entry = ttk.Entry(precision_frame, textvariable=self.precision_var, width=10)
        self.precision_entry.pack(side=tk.RIGHT, fill='x', expand=True, padx=(10, 0))

        # 5. Solve Button (Specification 5)
        self.solve_button = ttk.Button(self.input_frame, text="5. SOLVE SYSTEM", command=self.solve_system,
                                       style='Solve.TButton')
        self.solve_button.pack(fill='x', pady=(15, 0))

        # --- Output Frame (Right Side) ---
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Solution & Results", padding="10")
        self.output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.output_frame.columnconfigure(0, weight=1)

        ttk.Label(self.output_frame, text="Results Output:").pack(fill='x', pady=(0, 5))
        self.results_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=20, width=50,
                                                      font=("Consolas", 10), state=tk.DISABLED)
        self.results_text.pack(fill='both', expand=True)

        ttk.Label(self.output_frame, text="Details & Logs:").pack(fill='x', pady=(10, 5))
        self.log_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=5, width=50,
                                                  font=("Consolas", 9), state=tk.DISABLED)
        self.log_text.pack(fill='both', expand=True)

    def setup_styles(self):
        """Sets up custom styles for a modern look."""
        style = ttk.Style()
        style.theme_use('clam')  # Use a theme that supports custom styling

        # General widget style
        style.configure("TLabel", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10, "bold"), padding=6)
        style.configure("Solve.TButton", font=("Arial", 12, "bold"), foreground='white', background='#007BFF')
        style.map("Solve.TButton", background=[('active', '#0056b3')])
        style.configure("TEntry", padding=4)
        style.configure("TLabelframe", font=("Arial", 11, "bold"), foreground='#333333')

    def clear_params_frame(self):
        """Removes all widgets from the parameters frame."""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

    def update_parameters_frame(self, *args):
        """
        Dynamically updates the parameter fields based on the selected method.
        (Specification 3)
        """
        self.clear_params_frame()
        method = self.method_var.get()

        if method == "LU Decomposition":
            ttk.Label(self.params_frame, text="LU Form:").pack(fill='x', pady=(0, 5))
            lu_options = ["Doolittle Form", "Crout Form", "Cholesky Form"]
            ttk.Combobox(self.params_frame,
                         textvariable=self.lu_form_var,
                         values=lu_options,
                         state="readonly").pack(fill='x', pady=(0, 10))

        elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            # Initial Guess Input
            ttk.Label(self.params_frame, text="Initial Guess (comma-separated):").pack(fill='x', pady=(0, 5))
            ttk.Entry(self.params_frame, textvariable=self.initial_guess_var).pack(fill='x', pady=(0, 10))

            # Stopping Condition Type
            ttk.Label(self.params_frame, text="Stopping Condition Type:").pack(fill='x', pady=(0, 5))
            stop_type_options = ["Number of Iterations", "Absolute Relative Error"]
            ttk.Combobox(self.params_frame,
                         textvariable=self.stop_condition_type_var,
                         values=stop_type_options,
                         state="readonly").pack(fill='x', pady=(0, 10))

            # Stopping Value
            ttk.Label(self.params_frame, text="Stopping Value (e.g., Max Iterations or Error %):").pack(fill='x',
                                                                                                        pady=(0, 5))
            ttk.Entry(self.params_frame, textvariable=self.stop_value_var).pack(fill='x')

    def parse_guess_input(self, guess_str: str) -> Optional[List[float]]:
        """Parses the comma-separated initial guess string into a list of floats."""
        try:
            parts = [p.strip() for p in guess_str.split(',') if p.strip()]
            return [float(p) for p in parts]
        except ValueError:
            messagebox.showerror("Input Error", "Initial Guess must be a comma-separated list of numbers.")
            return None

    def get_user_params(self) -> Dict[str, Any]:
        """Collects all dynamic parameters based on the currently selected method."""
        method = self.method_var.get()
        params = {}

        if method == "LU Decomposition":
            params["LU Form"] = self.lu_form_var.get()

        elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            guess_list = self.parse_guess_input(self.initial_guess_var.get())
            if guess_list is None:
                # Raise an exception or handle error if initial guess is invalid
                return {"error": "Invalid Initial Guess Format"}

            params["Initial Guess"] = guess_list
            params["Stopping Condition Type"] = self.stop_condition_type_var.get()

            try:
                stop_value = float(self.stop_value_var.get())
                params["Stopping Value"] = stop_value
            except ValueError:
                return {"error": "Stopping Value must be a number."}

        return params

    def update_results_display(self, text: str, log: str):
        """Helper to safely update the ScrolledText widgets."""
        for widget in [self.results_text, self.log_text]:
            widget.config(state=tk.NORMAL)
            widget.delete(1.0, tk.END)

        self.results_text.insert(tk.END, text)
        self.log_text.insert(tk.END, log)

        for widget in [self.results_text, self.log_text]:
            widget.config(state=tk.DISABLED)

    def solve_system(self):
        """Handles the main logic when the Solve button is clicked."""
        self.update_results_display("Solving...", "Processing input and calculating...")

        # 1. Get and Validate Precision (Specification 4)
        try:
            precision = int(self.precision_var.get() or 5)  # Default to 5 if empty
            if precision <= 0 or precision > 15:
                raise ValueError("Precision must be a positive integer (max 15).")
            self.solver.set_precision(precision)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            self.update_results_display("", f"ERROR: {e}")
            return

        # 2. Get and Validate System Input (Specification 1)
        raw_input = self.input_text.get(1.0, tk.END)
        parsed_matrix = self.solver.parse_input(raw_input)

        if parsed_matrix is None:
            # Error handled inside parse_input via messagebox
            self.update_results_display("", "ERROR: System input validation failed.")
            return

        A, b = parsed_matrix

        # 3. Get Method Parameters (Specification 3)
        params = self.get_user_params()
        if "error" in params:
            messagebox.showerror("Input Error", params["error"])
            self.update_results_display("", f"ERROR: {params['error']}")
            return

        # --- 4. Call Solver ---
        method = self.method_var.get()

        # Check if initial guess size matches system size for iterative methods
        if method in ["Jacobi-Iteration", "Gauss-Seidel"] and len(params.get("Initial Guess", [])) != len(A):
            messagebox.showerror("Input Error",
                                 f"Initial guess must have {len(A)} components, matching the number of variables.")
            self.update_results_display("", "ERROR: Initial guess size mismatch.")
            return

        try:
            results = self.solver.solve(method, A, b, params)
        except Exception as e:
            results = {
                "success": False,
                "error_message": f"Critical Failure during solving: {e}",
                "execution_time": 0.0,
            }

        # --- 5. Display Results (Specifications 5, 6, 7) ---
        if results["success"]:
            sol_text = ""
            for i, val in enumerate(results["solution"]):
                # Format solution to the requested precision
                # Note: f-string formatting handles significant figures loosely;
                # for true significant figures, you need more complex formatting logic.
                # We'll use simple rounding here.
                formatted_val = f"{val:.{precision}f}".rstrip('0').rstrip('.')
                sol_text += f"X{i + 1} = {formatted_val}\n"

            output_text = f"--- Solution ---\n\n{sol_text}"

            log_text = (
                f"Method Used: {results['method_used']}\n"
                f"Precision (Sig Figs): {results['precision']}\n"
                f"Execution Time: {results['execution_time']:.6f} seconds\n"
            )

            if results["iterations"] != "N/A":
                log_text += f"Iterations Taken: {results['iterations']}\n"

            # Log parameters used
            log_text += "\nParameters Used:\n"
            log_text += json.dumps(params, indent=2)

        else:
            # No solution or infinite solutions / error (Specification 6)
            output_text = (
                f"--- Result ---\n\n"
                f"SYSTEM ERROR:\n"
                f"{results.get('error_message', 'The system could not be solved.')}"
            )
            log_text = (
                f"Method Used: {method}\n"
                f"Execution Time: {results.get('execution_time', 0.0):.6f} seconds\n"
                f"Input Data:\n"
                f"A = {A}\n"
                f"b = {b}\n"
            )

        self.update_results_display(output_text, log_text)


if __name__ == '__main__':
    # Set up the main window
    root = tk.Tk()
    app = NumericalSolverGUI(root)
    # Start the Tkinter event loop
    root.mainloop()