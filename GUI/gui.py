import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import json
# Import necessary type hints
from typing import List, Tuple, Dict, Any, Optional
import copy  # Needed for safe matrix copying
from NumericalSolver import NumericalSolver
from System.SystemData import SystemData


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # --- Canvas ---
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)

        # --- Scrollbars ---
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)

        # --- Frame inside canvas ---
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Place the inner frame inside the canvas
        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # --- Configure scroll region dynamically ---
        def update_scrollregion(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.scrollable_frame.bind("<Configure>", update_scrollregion)

        # --- Connect canvas to scrollbars ---
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        # --- Layout ---
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        # Make the frame expandable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # --- Optional: mouse wheel support for vertical scrolling ---
        self.scrollable_frame.bind(
            "<Enter>",
            lambda e: self.canvas.bind_all("<MouseWheel>",
                                           lambda ev: self.canvas.yview_scroll(int(-ev.delta / 120), "units"))
        )
        self.scrollable_frame.bind(
            "<Leave>",
            lambda e: self.canvas.unbind_all("<MouseWheel>")
        )


class NumericalSolverGUI:
    """
    The main Tkinter application window responsible for the interactive GUI
    (Specifications 1-5).
    """

    def __init__(self, master):
        self.master = master
        master.title("Numerical Linear System Solver (Project Phase 1)")

        # FIX: The previous error occurred because an imported module was treated as a class.
        # Since we cannot modify your external import (e.g., from 'Solver' module),
        # we ensure that the placeholder class 'NumericalSolver' is correctly
        # instantiated here, which is the correct pattern.
        self.solver = NumericalSolver()

        # --- Tkinter Variables for inputs ---
        self.method_var = tk.StringVar(master, value="Gauss Elimination")
        self.precision_var = tk.StringVar(master, value="5")  # Default precision (Spec 4)
        self.n_var = tk.StringVar(master, value="3")  # Default N=3 (Number of variables)

        # Dynamic parameter variables, initialized with defaults
        self.lu_form_var = tk.StringVar(master, value="Doolittle")
        self.initial_guess_var = tk.StringVar(master, value="0, 0, 0")  # Example for 3x3
        self.max_iter_var = tk.StringVar(master, value=100)
        self.error_tol_var = tk.StringVar(master, value=0.01)


        # Storage for the dynamically created matrix input widgets
        self.matrix_entry_widgets: List[List[tk.Entry]] = []

        # --- Setup Appearance ---
        self.setup_styles()

        # --- Setup Main Frames (Two-column layout) ---
        # Increased padding for a cleaner, less cramped look
        self.main_frame = ttk.Frame(master, padding="20 20 20 20", style='Main.TFrame')
        self.main_frame.pack(fill='both', expand=True)
        self.main_frame.columnconfigure(0, weight=1)  # Left input column
        self.main_frame.columnconfigure(1, weight=1)  # Right output column

        # --- Input Frame (Left Side) ---
        self.scroll_frame = ScrollableFrame(self.main_frame)
        self.scroll_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.scroll_frame.scrollable_frame.columnconfigure(0, weight=1)

        self.input_frame = ttk.LabelFrame(self.scroll_frame.scrollable_frame,
                                          text="1. System Input & Method Selection",
                                          padding="15",
                                          style='Input.TLabelframe')
        self.input_frame.pack(fill='both', expand=True)

        self.input_frame.columnconfigure(0, weight=1)

        # --- N Input, Matrix Generation Block, and Solve Button (Combined) ---
        # Changed the structure to group N input, Generate, and Solve buttons
        n_frame = ttk.Frame(self.input_frame)
        n_frame.pack(fill='x', pady=(10, 10))  # Adjusted top padding

        # N Input controls (packed left)
        ttk.Label(n_frame, text="N (Variables/Equations):", style='Title.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        self.n_entry = ttk.Entry(n_frame, textvariable=self.n_var, width=5, font=('Arial', 10))
        self.n_entry.pack(side=tk.LEFT, padx=(0, 15))

        # Generate Matrix Button (packed left)
        ttk.Button(n_frame, text="Generate Matrix", command=self.generate_matrix_input, style='Small.TButton').pack(
            side=tk.LEFT, padx=(0, 20))

        # 5. Solve Button (New position, packed left, beside Generate Matrix)
        self.solve_button = ttk.Button(n_frame, text="SOLVE SYSTEM", command=self.solve_system,
                                       style='InlineSolve.TButton')
        self.solve_button.pack(side=tk.LEFT)

        # Container frame for the dynamic matrix grid
        ttk.Label(self.input_frame, text="Enter Coefficients [A|b]:", style='Title.TLabel').pack(fill='x', pady=(10, 5))
        self.matrix_input_container = ttk.Frame(self.input_frame, style='Matrix.TFrame')
        self.matrix_input_container.pack(fill='x', expand=False, pady=(0, 15))

        # Initial draw of the 3x3 matrix input grid
        self.generate_matrix_input()

        # 2. Method Selection (Specification 2)
        ttk.Label(self.input_frame, text="2. Choose Solving Method:", style='Title.TLabel').pack(fill='x', pady=(10, 5))
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
                                            style='TCombobox',
                                            font=('Arial', 10))
        self.method_dropdown.pack(fill='x', pady=(0, 15))
        # Trigger dynamic parameter update whenever the method changes
        self.method_var.trace_add("write", self.update_parameters_frame)

        # 3. Dynamic Parameters Frame (Specification 3)
        self.params_frame = ttk.LabelFrame(self.input_frame, text="3. Method Parameters", padding="15",
                                           style='Input.TLabelframe')
        self.params_frame.pack(fill='x', pady=(10, 15))
        self.update_parameters_frame()  # Initial call to display default parameters

        # 4. Precision Input (Specification 4)
        precision_frame = ttk.Frame(self.input_frame)
        precision_frame.pack(fill='x', pady=(5, 10))
        ttk.Label(precision_frame, text="Precision (Significant Figures):", style='Title.TLabel').pack(side=tk.LEFT)
        self.precision_entry = ttk.Entry(precision_frame, textvariable=self.precision_var, width=10, style='TEntry',
                                         font=('Arial', 10))
        self.precision_entry.pack(side=tk.RIGHT, padx=(10, 0))

        # NOTE: The solve button placement (lines 354-356) has been moved and removed from here.

        # --- Output Frame (Right Side) ---
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Solution & Results", padding="15",
                                           style='Output.TLabelframe')
        self.output_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.output_frame.columnconfigure(0, weight=1)

        # Output area for the solution vector (X1, X2, ...)
        ttk.Label(self.output_frame, text="Results Output:", style='Title.TLabel').pack(fill='x', pady=(0, 5))
        self.results_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=20, width=50,
                                                      font=("Consolas", 11), state=tk.DISABLED, bg="#F0F8FF",
                                                      fg="#004D99",
                                                      relief=tk.FLAT)  # Flat relief for modern look
        self.results_text.pack(fill='both', expand=True)

        # Output area for execution time, iterations, and parameters (Specification 7)
        ttk.Label(self.output_frame, text="Details & Logs:", style='Title.TLabel').pack(fill='x', pady=(15, 5))
        self.log_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=5, width=50,
                                                  font=("Consolas", 10), state=tk.DISABLED, bg="#F5F5F5", fg="#555555",
                                                  relief=tk.FLAT)  # Flat relief for modern look
        self.log_text.pack(fill='both', expand=True)

    def setup_styles(self):
        """Sets up custom styles for a flat, modern aesthetic using the 'clam' theme as a base."""
        style = ttk.Style()
        # Use a more neutral modern base theme
        style.theme_use('clam')

        # --- Color Palette (Flat Design) ---
        PRIMARY_ACCENT = "#3498DB"  # Bright Blue for accents/primary actions
        SECONDARY_ACCENT = "#2ECC71"  # Green for the solve button/success
        BACKGROUND_LIGHT = "#ECF0F1"  # Very light grey background
        BACKGROUND_FRAME = "#FFFFFF"  # Pure white for contained elements
        TEXT_DARK = "#2C3E50"  # Dark slate grey for primary text
        TEXT_MEDIUM = "#7F8C8D"  # Medium grey for secondary text

        # --- General Styles ---
        style.configure("Main.TFrame", background=BACKGROUND_LIGHT)

        # --- Labels and Titles (Clean, consistent font) ---
        style.configure("TLabel", font=("Arial", 10), background=BACKGROUND_FRAME, foreground=TEXT_DARK)
        style.configure("Title.TLabel", font=("Arial", 11, "bold"), foreground=PRIMARY_ACCENT,
                        background=BACKGROUND_FRAME)

        # --- Label Frames (Containers) ---
        # Set background to white for content clarity, border to primary blue
        style.configure("Input.TLabelframe", font=("Arial", 12, "bold"), foreground=TEXT_DARK,
                        background=BACKGROUND_FRAME, bordercolor=PRIMARY_ACCENT)
        style.configure("Input.TLabelframe.Label", background=BACKGROUND_FRAME, foreground=PRIMARY_ACCENT,
                        bordercolor=PRIMARY_ACCENT)

        style.configure("Output.TLabelframe", font=("Arial", 12, "bold"), foreground=TEXT_DARK,
                        background=BACKGROUND_FRAME, bordercolor=SECONDARY_ACCENT)
        style.configure("Output.TLabelframe.Label", background=BACKGROUND_FRAME, foreground=SECONDARY_ACCENT,
                        bordercolor=SECONDARY_ACCENT)

        # --- Buttons (Flat and Primary Colors) ---
        style.configure("TButton", font=("Arial", 10, "bold"), padding=[10, 5], background=PRIMARY_ACCENT,
                        foreground='white', relief=tk.FLAT)
        style.configure("Small.TButton", font=("Arial", 9), padding=[8, 4], background="#BDC3C7", foreground=TEXT_DARK,
                        relief=tk.FLAT)

        # Solve Button (Original Full-Width Style - Kept for reference, but not used in the new location)
        style.configure("Solve.TButton", font=("Arial", 12, "bold"), foreground='white', background=SECONDARY_ACCENT,
                        padding=[15, 8], relief=tk.FLAT)
        style.map("Solve.TButton",
                  background=[('active', '#27AE60'), ('!disabled', SECONDARY_ACCENT)],
                  foreground=[('active', 'white'), ('!disabled', 'white')])

        # New style for inline solve button (reduced padding and font size for horizontal placement)
        style.configure("InlineSolve.TButton", font=("Arial", 10, "bold"), foreground='white',
                        background=SECONDARY_ACCENT,
                        padding=[10, 5], relief=tk.FLAT)
        style.map("InlineSolve.TButton",
                  background=[('active', '#27AE60'), ('!disabled', SECONDARY_ACCENT)],
                  foreground=[('active', 'white'), ('!disabled', 'white')])

        # --- Entries and Comboboxes (Flat borders) ---
        style.configure("TEntry", padding=[5, 5], background='white', bordercolor=TEXT_MEDIUM, fieldbackground='white',
                        foreground=TEXT_DARK)
        style.configure("TCombobox", padding=[5, 5], background='white', bordercolor=TEXT_MEDIUM,
                        fieldbackground='white', foreground=TEXT_DARK)

        # Apply a simple border/relief for input clarity
        style.configure("Matrix.TFrame", background=BACKGROUND_FRAME)
        style.configure("TFrame", background=BACKGROUND_FRAME)

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
            if N <= 0:
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
                      foreground="#2980B9").grid(row=0, column=j, padx=4, pady=4)
        ttk.Label(self.matrix_input_container, text=" | B", font=("Arial", 10, "bold"), foreground="#E74C3C").grid(
            row=0, column=N, padx=8, pady=4)

        # Create N rows and N+1 columns of Entry fields
        for i in range(N):
            row_entries = []
            for j in range(N + 1):
                entry = ttk.Entry(self.matrix_input_container, width=5, style='TEntry', font=('Consolas', 10))

                if j == N:
                    # Constant vector (B) column styling
                    entry.grid(row=i + 1, column=j, padx=(10, 2), pady=2, sticky='ew')
                    entry.config(foreground="#E74C3C")  # Red for constant vector
                else:
                    # Coefficient matrix (A) column styling
                    entry.grid(row=i + 1, column=j, padx=2, pady=2, sticky='ew')
                    entry.config(foreground="#2980B9")  # Blue for coefficients

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
            ttk.Label(self.params_frame, text="LU Form:", style='TLabel').pack(fill='x', pady=(5, 5))
            lu_options = ["Doolittle", "Crout", "Cholesky"]
            ttk.Combobox(self.params_frame,
                         textvariable=self.lu_form_var,
                         values=lu_options,
                         state="readonly",
                         style='TCombobox',
                         font=('Arial', 10)).pack(fill='x', pady=(0, 10))

        elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            # Requires initial guess and stopping condition

            # Initial Guess Input
            ttk.Label(self.params_frame, text="Initial Guess (comma-separated):", style='TLabel').pack(fill='x',
                                                                                                       pady=(5, 5))
            ttk.Entry(self.params_frame, textvariable=self.initial_guess_var, style='TEntry', font=('Arial', 10)).pack(
                fill='x', pady=(0, 10))


            # Stopping Conditions Title
            (ttk.Label(self.params_frame, text="Stopping Conditions:", style='Title.TLabel')
             .pack(fill='x', pady=(10, 5)))

            # --- Max Iterations ---
            (ttk.Label(self.params_frame, text="Max Iterations:", style='TLabel')
             .pack(fill='x', pady=(5, 2)))

            (ttk.Entry(self.params_frame, textvariable=self.max_iter_var, style='TEntry', font=('Arial', 10))
             .pack(fill='x'))

            # --- Error Tolerance (%) ---
            (ttk.Label(self.params_frame,text="Error Tolerance (%):",style='TLabel')
             .pack(fill='x', pady=(5, 2)))

            (ttk.Entry(self.params_frame, textvariable=self.error_tol_var, style='TEntry', font=('Arial', 10))
             .pack(fill='x'))

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


            # Validate Stopping Value format
            try:
                params["max_iter_var"] = int(self.max_iter_var.get())
                params["error_tol_var"] = float(self.error_tol_var.get())
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
        # Uses the parse_input method from the NumericalSolver placeholder
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

        # if method in ["Jacobi-Iteration", "Gauss-Seidel"]:

        # --- 5. Create DTO and Call Solver ---
        # Pass deep copies of A and b to ensure the solver works on its own version
        system_data = SystemData(copy.deepcopy(A), copy.deepcopy(b), method, precision, params)
        self.update_results_display("Solving...", f"Dispatching to {method} Solver...")

        try:
            # Dispatch DTO to the solver entry point
            # Uses the solve method from the NumericalSolver placeholder
            results = self.solver.solve(system_data)
        except Exception as e:
            # Catch unexpected errors during the Factory/Solver process
            results = {
                "success": False,
                "error_message": f"Critical Failure during solving: {e}",
                "execution_time": 0.0,
            }

        # --- 6. Display Results (Specifications 5, 6, 7) ---
        # if results["success"]:
        sol_text = ""
        # Format the solution based on the requested precision
        for i, val in enumerate(results["sol"]):
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

        # else:
        #     # Display error message for no solution, infinite solutions, or unexpected error (Specification 6)
        # output_text = (
        #     f"--- Result ---\n\n"
        #     f"SYSTEM ERROR:\n"
        #     f"{results.get('error_message', 'The system could not be solved.')}"
        # )
        # log_text = (
        #     f"Method Used: {method}\n"
        #     f"Execution Time: {results.get('execution_time', 0.0):.6f} seconds\n"
        #     f"Input Data Size: {N}x{N}\n"
        # )

        self.update_results_display(output_text, log_text)


if __name__ == '__main__':
    # Debug print to confirm script is running
    print("Starting Numerical Solver GUI application...")
    # Set up the main window
    root = tk.Tk()
    app = NumericalSolverGUI(root)
    # Start the Tkinter event loop, which handles all GUI interactions
    root.mainloop()