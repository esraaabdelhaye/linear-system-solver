import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
import time
import json
# Used for type hinting for cleaner, more readable code
from typing import List, Tuple, Dict, Any, Optional
import copy  # Needed for safe matrix copying


# --- 1. DATA TRANSFER OBJECT (DTO) ---

class SystemData:
    """
    Data Transfer Object (DTO) for passing system configuration and data
    from the GUI layer to the Solver layer (clean separation of concerns).
    """

    def __init__(self, A: List[List[float]], b: List[float], method: str,
                 precision: int, params: Dict[str, Any]):
        self.A = A  # Coefficient Matrix (2D list of floats)
        self.b = b  # Constant Vector (1D list of floats)
        self.method = method  # Solving method (e.g., "Gauss Elimination")
        self.precision = precision  # Number of significant figures (Specification 4)
        self.params = params  # Method-specific parameters (e.g., initial guess, LU form)
        self.N = len(A)  # Size of the system (Number of Variables/Equations)


# --- 2. SOLVER INTERFACE AND FACTORY ---

class BaseSolver:
    """
    Base class (Interface) for all numerical solving methods.
    All specific solvers must inherit from this and implement the solve method.
    """

    def __init__(self, data: SystemData):
        self.data = data
        self.A = data.A
        self.b = data.b
        self.N = data.N
        self.precision = data.precision

    def solve(self) -> Dict[str, Any]:
        """
        Abstract method to be implemented by derived classes.
        Must return a dictionary containing 'success', 'solution', 'execution_time', and 'iterations'.
        """
        raise NotImplementedError("Subclasses must implement the solve method.")


# --- Specific Solver Implementations ---

class GaussEliminationSolver(BaseSolver):
    """
    Implements the Gauss Elimination method with mandatory partial pivoting
    (Specification 8) to solve Ax = b.
    """

    def solve(self) -> Dict[str, Any]:
        start_time = time.time()

        N = self.N
        # Create an augmented matrix [A|b] for in-place modification
        Aug = [self.A[i] + [self.b[i]] for i in range(N)]

        # --- 1. Forward Elimination with Partial Pivoting ---
        for i in range(N):
            # 1a. Partial Pivoting: Find the row with the largest absolute value in the current column
            max_val = abs(Aug[i][i])
            max_row = i
            for k in range(i + 1, N):
                if abs(Aug[k][i]) > max_val:
                    max_val = abs(Aug[k][i])
                    max_row = k

            # Swap the current row (i) with the max row (max_row) if a larger pivot is found
            if max_row != i:
                Aug[i], Aug[max_row] = Aug[max_row], Aug[i]

            # 1b. Check for leading zero after pivoting (singular matrix)
            if abs(Aug[i][i]) < 1e-12:  # Use a small tolerance for zero check

                # Check for No Solution vs. Infinite Solutions (Specification 6)
                is_all_zero = True
                for j in range(i, N):
                    if abs(Aug[i][j]) >= 1e-12:
                        is_all_zero = False
                        break

                if is_all_zero and abs(Aug[i][N]) < 1e-12:
                    # Row of zeros [0 0 ... 0 | 0] means infinite solutions
                    raise ValueError("The system has infinite number of solutions.")
                elif is_all_zero and abs(Aug[i][N]) >= 1e-12:
                    # Row of zeros with a non-zero constant [0 0 ... 0 | C] means no solution
                    raise ValueError("The system has no solution (inconsistent).")
                else:
                    # System is singular but potentially solvable or ill-conditioned
                    raise ValueError(
                        "The system is singular or ill-conditioned, and cannot be solved with this method.")

            # 1c. Eliminate below the pivot
            for k in range(i + 1, N):
                factor = Aug[k][i] / Aug[i][i]
                for j in range(i, N + 1):
                    # Aug[k][j] = Aug[k][j] - factor * Aug[i][j]
                    # We subtract slightly modified versions of the pivot row from the rows below
                    Aug[k][j] -= factor * Aug[i][j]

        # --- 2. Back Substitution ---
        solution = [0.0] * N
        for i in range(N - 1, -1, -1):
            # Start with the constant term (Aug[i][N])
            sum_of_knowns = Aug[i][N]

            # Subtract the terms involving already solved variables (X_{i+1} to X_{N})
            for j in range(i + 1, N):
                sum_of_knowns -= Aug[i][j] * solution[j]

            # Solve for the current variable (X_i)
            solution[i] = sum_of_knowns / Aug[i][i]

        execution_time = time.time() - start_time
        return {
            "success": True,
            "solution": solution,
            "execution_time": execution_time,
            "iterations": "N/A (Direct Method)",
        }


class GaussJordanSolver(BaseSolver):
    """Placeholder for the Gauss-Jordan method logic."""

    def solve(self) -> Dict[str, Any]:
        # Implementation will be similar to Gauss Elimination but requires
        # elimination both above and below the diagonal.
        raise NotImplementedError("Gauss-Jordan Solver not yet implemented.")


class LUSolver(BaseSolver):
    """Placeholder for the LU Decomposition method logic."""

    def solve(self) -> Dict[str, Any]:
        # This will need to check self.data.params['LU Form']
        # It must also apply Partial Pivoting (Specification 8) if Doolittle/Crout form is used.
        raise NotImplementedError("LU Decomposition Solver not yet implemented.")


class JacobiSolver(BaseSolver):
    """Placeholder for the Jacobi Iteration method logic."""

    def solve(self) -> Dict[str, Any]:
        start_time = time.time()
        # --- YOUR JACOBI ITERATION LOGIC HERE ---
        # Use self.data.params for 'Initial Guess', 'Stopping Condition Type', 'Stopping Value'

        time.sleep(1.0)  # Simulate calculation time

        # Placeholder result
        solution = [float(i) + 0.5 for i in range(self.N)]
        iterations = 42  # Example iteration count

        # Example check for convergence (Specification 6)
        # if not is_diagonally_dominant(self.A):
        #    raise ValueError("System may not converge (not diagonally dominant).")

        execution_time = time.time() - start_time
        return {
            "success": True,
            "solution": solution,
            "execution_time": execution_time,
            "iterations": iterations,
        }


class GaussSeidelSolver(BaseSolver):
    """Placeholder for the Gauss-Seidel method logic."""

    def solve(self) -> Dict[str, Any]:
        # Implementation will be similar to Jacobi but uses newly computed
        # values immediately in the same iteration.
        raise NotImplementedError("Gauss-Seidel Solver not yet implemented.")


class SolverFactory:
    """
    Factory class to instantiate the correct solver based on the method
    specified in the SystemData DTO. This decouples the GUI from specific solvers.
    """
    SOLVERS = {
        "Gauss Elimination": GaussEliminationSolver,
        "Gauss-Jordan": GaussEliminationSolver,  # Placeholder, should be GaussJordanSolver
        "LU Decomposition": GaussEliminationSolver,  # Placeholder, should be LUSolver
        "Jacobi-Iteration": JacobiSolver,
        "Gauss-Seidel": JacobiSolver,  # Placeholder, should be GaussSeidelSolver
    }

    @staticmethod
    def get_solver(data: SystemData) -> BaseSolver:
        """Returns an instance of the specific solver class."""
        solver_class = SolverFactory.SOLVERS.get(data.method)
        if not solver_class:
            raise ValueError(f"Solver for method '{data.method}' is not implemented.")
        # Instantiate the correct solver with the DTO
        return solver_class(data)


# --- 3. NUMERICAL SOLVER CLASS (Refactored to use Factory) ---

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


# --- 4. GUI APPLICATION CLASS (Refactored) ---

class NumericalSolverGUI:
    """
    The main Tkinter application window responsible for the interactive GUI
    (Specifications 1-5).
    """

    def __init__(self, master):
        self.master = master
        master.title("Numerical Linear System Solver (Project Phase 1)")

        # Initialize the coordinator backend
        self.solver = NumericalSolver()

        # --- Tkinter Variables for inputs ---
        self.method_var = tk.StringVar(master, value="Gauss Elimination")
        self.precision_var = tk.StringVar(master, value="5")  # Default precision (Spec 4)
        self.n_var = tk.StringVar(master, value="3")  # Default N=3 (Number of variables)

        # Dynamic parameter variables, initialized with defaults
        self.lu_form_var = tk.StringVar(master, value="Doolittle Form")
        self.initial_guess_var = tk.StringVar(master, value="0, 0, 0")  # Example for 3x3
        self.stop_condition_type_var = tk.StringVar(master, value="Number of Iterations")
        self.stop_value_var = tk.StringVar(master, value="50")

        # Storage for the dynamically created matrix input widgets
        self.matrix_entry_widgets: List[List[tk.Entry]] = []

        # --- Setup Appearance ---
        self.setup_styles()

        # --- Setup Main Frames (Two-column layout) ---
        self.main_frame = ttk.Frame(master, padding="15 15 15 15", style='Main.TFrame')
        self.main_frame.pack(fill='both', expand=True)
        self.main_frame.columnconfigure(0, weight=1)  # Left input column
        self.main_frame.columnconfigure(1, weight=1)  # Right output column

        # --- Input Frame (Left Side) ---
        self.input_frame = ttk.LabelFrame(self.main_frame, text="1. System Input & Method Selection", padding="10",
                                          style='Input.TLabelframe')
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.input_frame.columnconfigure(0, weight=1)

        # --- N Input and Matrix Generation Block ---
        n_frame = ttk.Frame(self.input_frame)
        n_frame.pack(fill='x', pady=(0, 5))
        ttk.Label(n_frame, text="N (Variables/Equations):", style='Title.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        self.n_entry = ttk.Entry(n_frame, textvariable=self.n_var, width=5)
        self.n_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(n_frame, text="Generate Matrix", command=self.generate_matrix_input, style='Small.TButton').pack(
            side=tk.LEFT)

        # Container frame for the dynamic matrix grid
        ttk.Label(self.input_frame, text="Enter Coefficients [A|b]:", style='Title.TLabel').pack(fill='x', pady=(5, 5))
        self.matrix_input_container = ttk.Frame(self.input_frame, style='Matrix.TFrame')
        self.matrix_input_container.pack(fill='x', expand=False, pady=(0, 10))

        # Initial draw of the 3x3 matrix input grid
        self.generate_matrix_input()

        # 2. Method Selection (Specification 2)
        ttk.Label(self.input_frame, text="2. Choose Solving Method:", style='Title.TLabel').pack(fill='x', pady=(5, 5))
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
                                            state="readonly",
                                            style='TCombobox')
        self.method_dropdown.pack(fill='x', pady=(0, 10))
        # Trigger dynamic parameter update whenever the method changes
        self.method_var.trace_add("write", self.update_parameters_frame)

        # 3. Dynamic Parameters Frame (Specification 3)
        self.params_frame = ttk.LabelFrame(self.input_frame, text="3. Method Parameters", padding="10",
                                           style='Input.TLabelframe')
        self.params_frame.pack(fill='x', pady=(5, 10))
        self.update_parameters_frame()  # Initial call to display default parameters

        # 4. Precision Input (Specification 4)
        precision_frame = ttk.Frame(self.input_frame)
        precision_frame.pack(fill='x', pady=(5, 5))
        ttk.Label(precision_frame, text="4. Precision (Significant Figures):", style='Title.TLabel').pack(side=tk.LEFT)
        self.precision_entry = ttk.Entry(precision_frame, textvariable=self.precision_var, width=10, style='TEntry')
        self.precision_entry.pack(side=tk.RIGHT, padx=(10, 0))

        # 5. Solve Button (Specification 5)
        self.solve_button = ttk.Button(self.input_frame, text="5. SOLVE SYSTEM", command=self.solve_system,
                                       style='Solve.TButton')
        self.solve_button.pack(fill='x', pady=(15, 0))

        # --- Output Frame (Right Side) ---
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Solution & Results", padding="10",
                                           style='Output.TLabelframe')
        self.output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.output_frame.columnconfigure(0, weight=1)

        # Output area for the solution vector (X1, X2, ...)
        ttk.Label(self.output_frame, text="Results Output:", style='Title.TLabel').pack(fill='x', pady=(0, 5))
        self.results_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=20, width=50,
                                                      font=("Consolas", 10), state=tk.DISABLED, bg="#f0f0f0",
                                                      fg="#000080")
        self.results_text.pack(fill='both', expand=True)

        # Output area for execution time, iterations, and parameters (Specification 7)
        ttk.Label(self.output_frame, text="Details & Logs:", style='Title.TLabel').pack(fill='x', pady=(10, 5))
        self.log_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=5, width=50,
                                                  font=("Consolas", 9), state=tk.DISABLED, bg="#f0f0f0", fg="#333333")
        self.log_text.pack(fill='both', expand=True)

    def setup_styles(self):
        """Sets up custom styles for a modern, colorful look using the 'clam' theme."""
        style = ttk.Style()
        style.theme_use('clam')

        # Color Palette definition
        PRIMARY_BLUE = "#007BFF"  # Accent color for input titles
        SECONDARY_GREEN = "#28A745"  # Accent color for the solve button and output frame
        BACKGROUND_LIGHT = "#f8f9fa"
        BACKGROUND_DARK = "#e9ecef"
        TEXT_DARK = "#343A40"

        # General Frame/Background
        style.configure("Main.TFrame", background=BACKGROUND_LIGHT)

        # Labels and Titles
        style.configure("TLabel", font=("Arial", 10), background=BACKGROUND_LIGHT, foreground=TEXT_DARK)
        style.configure("Title.TLabel", font=("Arial", 11, "bold"), foreground=PRIMARY_BLUE)

        # Label Frames (Containers)
        style.configure("Input.TLabelframe", font=("Arial", 12, "bold"), foreground=PRIMARY_BLUE,
                        background=BACKGROUND_DARK)
        style.configure("Input.TLabelframe.Label", background=BACKGROUND_DARK, foreground=PRIMARY_BLUE)
        style.configure("Output.TLabelframe", font=("Arial", 12, "bold"), foreground=SECONDARY_GREEN,
                        background=BACKGROUND_DARK)
        style.configure("Output.TLabelframe.Label", background=BACKGROUND_DARK, foreground=SECONDARY_GREEN)

        # Buttons
        style.configure("TButton", font=("Arial", 10), padding=6, background=BACKGROUND_LIGHT)
        style.configure("Small.TButton", font=("Arial", 9), padding=3, background='#D0D0D0')
        style.configure("Solve.TButton", font=("Arial", 12, "bold"), foreground='white', background=SECONDARY_GREEN)
        # Map ensures button color changes on interaction (active/hover)
        style.map("Solve.TButton", background=[('active', '#1E7E34'), ('!disabled', SECONDARY_GREEN)])

        # Entries
        style.configure("TEntry", padding=4, background='white')
        style.configure("Matrix.TFrame", background=BACKGROUND_LIGHT)

    def clear_params_frame(self):
        """Removes all widgets from the parameters frame to prepare for dynamic content."""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

    def generate_matrix_input(self):
        """
        Dynamically generates N x (N+1) Entry widgets for matrix input based on N.
        This enforces Specification 1c (N variables = N equations).
        """
        try:
            N = int(self.n_var.get())
            if N <= 0 or N > 10:
                messagebox.showwarning("Input Warning", "N must be between 1 and 10 for a usable layout.")
                self.n_var.set("3")  # Reset to default if out of range
                N = 3
        except ValueError:
            messagebox.showerror("Input Error", "N must be an integer.")
            self.n_var.set("3")
            return

        # Clear existing entries/widgets in the container
        for widget in self.matrix_input_container.winfo_children():
            widget.destroy()

        self.matrix_entry_widgets = []

        # Create Header Row (X1, X2, ..., | B)
        for j in range(N):
            ttk.Label(self.matrix_input_container, text=f"X{j + 1}", font=("Arial", 10, "bold"),
                      foreground="#007BFF").grid(row=0, column=j, padx=2, pady=2)
        ttk.Label(self.matrix_input_container, text=" | B", font=("Arial", 10, "bold"), foreground="#DC3545").grid(
            row=0, column=N, padx=5, pady=2)

        # Create N rows and N+1 columns of Entry fields
        for i in range(N):
            row_entries = []
            for j in range(N + 1):
                entry = ttk.Entry(self.matrix_input_container, width=5, style='TEntry')

                if j == N:
                    # Constant vector (B) column styling
                    entry.grid(row=i + 1, column=j, padx=(10, 2), pady=2, sticky='ew')
                    entry.config(foreground="#DC3545")  # Red for constant vector
                else:
                    # Coefficient matrix (A) column styling
                    entry.grid(row=i + 1, column=j, padx=2, pady=2, sticky='ew')
                    entry.config(foreground="#007BFF")  # Blue for coefficients

                row_entries.append(entry)
            self.matrix_entry_widgets.append(row_entries)

        # Populate a default 3x3 system for easy testing
        initial_data = [
            [4.0, 1.0, -1.0, 3.0],
            [2.0, 7.0, 1.0, 19.0],
            [1.0, -3.0, 12.0, 31.0]
        ]

        for i in range(min(N, 3)):  # Only fill up to N=3 for the initial example
            for j in range(N + 1):
                if i < len(initial_data) and j < len(initial_data[i]):
                    self.matrix_entry_widgets[i][j].delete(0, tk.END)
                    self.matrix_entry_widgets[i][j].insert(0, str(initial_data[i][j]))

    def update_parameters_frame(self, *args):
        """
        Dynamically updates the parameter input fields based on the selected method.
        (Specification 3)
        """
        self.clear_params_frame()
        method = self.method_var.get()

        if method == "LU Decomposition":
            # Requires LU form selection
            ttk.Label(self.params_frame, text="LU Form:", style='TLabel').pack(fill='x', pady=(0, 5))
            lu_options = ["Doolittle Form", "Crout Form", "Cholesky Form"]
            ttk.Combobox(self.params_frame,
                         textvariable=self.lu_form_var,
                         values=lu_options,
                         state="readonly",
                         style='TCombobox').pack(fill='x', pady=(0, 10))

        elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            # Requires initial guess and stopping condition

            # Initial Guess Input
            ttk.Label(self.params_frame, text="Initial Guess (comma-separated):", style='TLabel').pack(fill='x',
                                                                                                       pady=(0, 5))
            ttk.Entry(self.params_frame, textvariable=self.initial_guess_var, style='TEntry').pack(fill='x',
                                                                                                   pady=(0, 10))

            # Stopping Condition Type (Dropdown)
            ttk.Label(self.params_frame, text="Stopping Condition Type:", style='TLabel').pack(fill='x', pady=(0, 5))
            stop_type_options = ["Number of Iterations", "Absolute Relative Error"]
            ttk.Combobox(self.params_frame,
                         textvariable=self.stop_condition_type_var,
                         values=stop_type_options,
                         state="readonly",
                         style='TCombobox').pack(fill='x', pady=(0, 10))

            # Stopping Value (Entry)
            ttk.Label(self.params_frame, text="Stopping Value (e.g., Max Iterations or Error %):", style='TLabel').pack(
                fill='x', pady=(0, 5))
            ttk.Entry(self.params_frame, textvariable=self.stop_value_var, style='TEntry').pack(fill='x')

    def parse_guess_input(self, guess_str: str) -> Optional[List[float]]:
        """Parses the comma-separated initial guess string into a list of floats."""
        try:
            # Split by comma, strip spaces, filter empty parts, convert to float
            parts = [p.strip() for p in guess_str.split(',') if p.strip()]
            return [float(p) for p in parts]
        except ValueError:
            # Returns None if any part is not a valid number
            return None

    def get_user_params(self) -> Dict[str, Any]:
        """Collects all dynamic parameters from the GUI based on the selected method."""
        method = self.method_var.get()
        params = {}

        if method == "LU Decomposition":
            params["LU Form"] = self.lu_form_var.get()

        elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            # Store raw guess string first, validate size later in solve_system
            params["Initial Guess (Raw)"] = self.initial_guess_var.get()
            params["Stopping Condition Type"] = self.stop_condition_type_var.get()

            # Validate Stopping Value format
            try:
                stop_value = float(self.stop_value_var.get())
                params["Stopping Value"] = stop_value
            except ValueError:
                return {"error": "Stopping Value must be a number."}

        return params

    def update_results_display(self, text: str, log: str):
        """Helper to safely enable, clear, update, and disable ScrolledText widgets."""
        for widget in [self.results_text, self.log_text]:
            widget.config(state=tk.NORMAL)
            widget.delete(1.0, tk.END)

        self.results_text.insert(tk.END, text)
        self.log_text.insert(tk.END, log)

        for widget in [self.results_text, self.log_text]:
            widget.config(state=tk.DISABLED)

    def solve_system(self):
        """
        Main function executed when the 'SOLVE SYSTEM' button is pressed.
        Coordinates input validation, DTO creation, and calls the solver backend.
        """
        self.update_results_display("Solving...", "Processing input and creating DTO...")

        # 0. Get N (Number of Variables)
        try:
            N = int(self.n_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "N must be an integer.")
            self.update_results_display("", "ERROR: N input is invalid.")
            return

        # 1. Get and Validate Precision (Specification 4)
        try:
            precision = int(self.precision_var.get() or 5)  # Default to 5 sig figs
            if precision <= 0 or precision > 15:
                raise ValueError("Precision must be a positive integer (max 15).")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            self.update_results_display("", f"ERROR: {e}")
            return

        # 2. Get and Validate System Input (Specification 1)
        # Note: We pass a deep copy of A and b to prevent the solver from modifying the input data
        parsed_matrix = self.solver.parse_input(self.matrix_entry_widgets, N)

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

        # 4. Final Parameter Validation (Specific to Iterative methods)
        method = self.method_var.get()
        if method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            raw_guess = params.pop("Initial Guess (Raw)")  # Get raw string
            guess_list = self.parse_guess_input(raw_guess)  # Parse into float list

            if guess_list is None:
                messagebox.showerror("Input Error", "Initial Guess must be a comma-separated list of numbers.")
                self.update_results_display("", "ERROR: Invalid initial guess format.")
                return

            if len(guess_list) != N:
                messagebox.showerror("Input Error",
                                     f"Initial guess must have {N} components, matching the number of variables.")
                self.update_results_display("", "ERROR: Initial guess size mismatch.")
                return

            params["Initial Guess"] = guess_list  # Store validated list in params

        # --- 5. Create DTO and Call Solver ---
        # Pass deep copies of A and b to ensure the solver works on its own version
        system_data = SystemData(copy.deepcopy(A), copy.deepcopy(b), method, precision, params)
        self.update_results_display("Solving...", f"Dispatching to {method} Solver...")

        try:
            # Dispatch DTO to the solver entry point
            results = self.solver.solve(system_data)
        except Exception as e:
            # Catch unexpected errors during the Factory/Solver process
            results = {
                "success": False,
                "error_message": f"Critical Failure during solving: {e}",
                "execution_time": 0.0,
            }

        # --- 6. Display Results (Specifications 5, 6, 7) ---
        if results["success"]:
            sol_text = ""
            # Format the solution based on the requested precision
            for i, val in enumerate(results["solution"]):
                formatted_val = f"{val:.{precision}f}".rstrip('0').rstrip('.')
                sol_text += f"X{i + 1} = {formatted_val}\n"

            output_text = f"--- Solution ---\n\n{sol_text}"

            # Prepare logs
            log_params = {k: v for k, v in params.items()}

            log_text = (
                f"Method Used: {results['method_used']}\n"
                f"Precision (Sig Figs): {results['precision']}\n"
                f"Execution Time: {results['execution_time']:.6f} seconds\n"
            )

            if results.get("iterations") != "N/A":
                log_text += f"Iterations Taken: {results.get('iterations', 'N/A')}\n"

            # Display parameters used
            log_text += "\nParameters Used:\n"
            log_text += json.dumps(log_params, indent=2)

        else:
            # Display error message for no solution, infinite solutions, or unexpected error (Specification 6)
            output_text = (
                f"--- Result ---\n\n"
                f"SYSTEM ERROR:\n"
                f"{results.get('error_message', 'The system could not be solved.')}"
            )
            log_text = (
                f"Method Used: {method}\n"
                f"Execution Time: {results.get('execution_time', 0.0):.6f} seconds\n"
                f"Input Data Size: {N}x{N}\n"
            )

        self.update_results_display(output_text, log_text)


if __name__ == '__main__':
    # Debug print to confirm script is running
    print("Starting Numerical Solver GUI application...")
    # Set up the main window
    root = tk.Tk()
    app = NumericalSolverGUI(root)
    # Start the Tkinter event loop, which handles all GUI interactions
    root.mainloop()