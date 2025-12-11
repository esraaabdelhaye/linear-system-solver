"""
Numerical Solver GUI - Phase 1 & Phase 2 - COMPLETE
File: GUI/gui.py

GUI Application with two tabs:
- Phase 1: Linear System Solver (Gaussian Elimination, Gauss-Jordan, LU Decomposition, Iterative Methods)
           WITH BONUSES: Single-Step Mode (#1) & Scaling (#3)
- Phase 2: Root Finder (Bisection, False-Position, Fixed Point, Newton-Raphson, etc.)
           WITH BONUS: Single-Step Mode (#1)

Features:
- Bonus #1: Single-step mode simulation showing each step of the algorithm (BOTH PHASES)
- Bonus #3: Scaling as explained in the lecture (PHASE 1 ONLY)
"""

# ============================================================================
# IMPORTS
# ============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import List
import copy
import time
import re
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Phase 1 Imports
from NumericalSolver import NumericalSolver
from System.SystemData import SystemData

# Phase 2 Imports
from RootFinder import RootFinderData, RootFinderFactory

# Plotting Imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


# ============================================================================
# SCROLLABLE FRAME HELPER
# ============================================================================

class ScrollableFrame(ttk.Frame):
    """
    Custom frame widget with scrollbars for large content
    Used in both Phase 1 and Phase 2 input areas
    """

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def update_scrollregion(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.scrollable_frame.bind("<Configure>", update_scrollregion)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.scrollable_frame.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>",
                                                                             lambda ev: self.canvas.yview_scroll(
                                                                                 int(-ev.delta / 120), "units")))
        self.scrollable_frame.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class NumericalSolverGUI:
    """
    Main GUI Application with Phase 1 and Phase 2

    Features:
    - Two-tab interface for Phase 1 (Linear Systems) and Phase 2 (Root Finding)
    - Dynamic parameter updating based on selected method
    - Single-step mode bonus feature for BOTH phases
    - Scaling option bonus feature for Phase 1 direct methods
    - Comprehensive error handling and user feedback
    """

    def __init__(self, master):
        """
        Initialize the GUI application

        Args:
            master: Tkinter root window
        """
        self.master = master
        master.title("Numerical Solver - Phase 1 & 2")
        master.geometry("1400x900")

        # Initialize solver coordinator
        self.solver = NumericalSolver()

        # ====================================================================
        # PHASE 1 VARIABLES (Linear Systems)
        # ====================================================================
        self.method_var = tk.StringVar(master, value="Gauss Elimination")
        self.precision_var = tk.StringVar(master, value="5")
        self.n_var = tk.StringVar(master, value="3")
        self.lu_form_var = tk.StringVar(master, value="Doolittle")
        self.initial_guess_var = tk.StringVar(master, value="0, 0, 0")
        self.max_iter_var = tk.StringVar(master, value="100")
        self.error_tol_var = tk.StringVar(master, value="0.01")
        self.use_scaling_var = tk.BooleanVar(master, value=False)  # BONUS #3

        # Phase 1 Single-Step Mode Variables (Bonus #1)
        self.single_step_phase1_var = tk.BooleanVar(master, value=False)
        self.step_mode_active_phase1 = False
        self.current_step_index_phase1 = 0
        self.all_steps_phase1 = []
        self.execution_time_phase1 = 0.0

        # ====================================================================
        # PHASE 2 VARIABLES (Root Finding)
        # ====================================================================
        self.equation_var = tk.StringVar(master, value="x**2 - 4")
        self.root_method_var = tk.StringVar(master, value="Bisection")
        self.root_precision_var = tk.StringVar(master, value="5")
        self.epsilon_var = tk.StringVar(master, value="0.00001")
        self.max_iter_root_var = tk.StringVar(master, value="50")
        self.interval_a_var = tk.StringVar(master, value="-5")
        self.interval_b_var = tk.StringVar(master, value="5")
        self.initial_guess_root_var = tk.StringVar(master, value="1")
        self.second_guess_root_var = tk.StringVar(master, value="2")
        self.delta_var = tk.StringVar(master, value="0.01")
        self.single_step_var = tk.BooleanVar(value=False)

        # Phase 2 Step mode variables
        self.step_mode_active = False
        self.current_step_index = 0
        self.all_steps = []
        self.execution_time = 0.0

        # Matrix input widgets storage
        self.matrix_entry_widgets: List[List[tk.Entry]] = []

        # Setup styles and UI
        self.setup_styles()
        self.setup_notebook()

    def setup_styles(self):
        """Configure ttk styles for modern appearance"""
        style = ttk.Style()
        style.theme_use('clam')

        # Color palette
        PRIMARY_ACCENT = "#3498DB"
        SECONDARY_ACCENT = "#2ECC71"
        BACKGROUND_LIGHT = "#ECF0F1"
        BACKGROUND_FRAME = "#FFFFFF"
        TEXT_DARK = "#2C3E50"
        TEXT_MEDIUM = "#7F8C8D"

        # General styles
        style.configure("Main.TFrame", background=BACKGROUND_LIGHT)
        style.configure("TLabel", font=("Arial", 10), background=BACKGROUND_FRAME, foreground=TEXT_DARK)
        style.configure("Title.TLabel", font=("Arial", 11, "bold"), foreground=PRIMARY_ACCENT,
                        background=BACKGROUND_FRAME)
        style.configure("Input.TLabelframe", font=("Arial", 12, "bold"), foreground=TEXT_DARK,
                        background=BACKGROUND_FRAME)
        style.configure("Output.TLabelframe", font=("Arial", 12, "bold"), foreground=TEXT_DARK,
                        background=BACKGROUND_FRAME)
        style.configure("TButton", font=("Arial", 10, "bold"), padding=[10, 5], background=PRIMARY_ACCENT,
                        foreground='white', relief=tk.FLAT)
        style.configure("InlineSolve.TButton", font=("Arial", 10, "bold"), foreground='white',
                        background=SECONDARY_ACCENT, padding=[10, 5], relief=tk.FLAT)

    def setup_notebook(self):
        """Setup main notebook with Phase 1 and Phase 2 tabs"""
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Phase 1 Tab
        self.phase1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.phase1_frame, text="Phase 1: Linear Systems")
        self.setup_phase1_ui(self.phase1_frame)

        # Phase 2 Tab
        self.phase2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.phase2_frame, text="Phase 2: Root Finding")
        self.setup_phase2_ui(self.phase2_frame)

    def setup_phase1_ui(self, parent):
        """Setup Phase 1 Linear Systems UI with Scaling and Single-Step Bonuses"""
        main_frame = ttk.Frame(parent, padding="20 20 20 20")
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Input Frame (Left)
        scroll_frame = ScrollableFrame(main_frame)
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        scroll_frame.scrollable_frame.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(scroll_frame.scrollable_frame, text="System Input & Method Selection",
                                     padding="15")
        input_frame.pack(fill='both', expand=True)
        input_frame.columnconfigure(0, weight=1)

        # N Input
        n_frame = ttk.Frame(input_frame)
        n_frame.pack(fill='x', pady=(10, 10))
        ttk.Label(n_frame, text="N (Variables/Equations):", style='Title.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        self.n_entry = ttk.Entry(n_frame, textvariable=self.n_var, width=5, font=('Arial', 10))
        self.n_entry.pack(side=tk.LEFT, padx=(0, 15))
        ttk.Button(n_frame, text="Generate Matrix", command=self.generate_matrix_input).pack(side=tk.LEFT, padx=(0, 20))
        self.solve_button = ttk.Button(n_frame, text="SOLVE", command=self.solve_linear_system)
        self.solve_button.pack(side=tk.LEFT, padx=(0, 5))

        # Phase 1 Single-Step Navigation Buttons
        self.next_step_button_p1 = ttk.Button(n_frame, text="Next Step",
                                             command=self.next_step_phase1, state=tk.DISABLED)
        self.next_step_button_p1.pack(side=tk.LEFT, padx=(0, 5))
        self.reset_step_button_p1 = ttk.Button(n_frame, text="Reset",
                                              command=self.reset_step_phase1, state=tk.DISABLED)
        self.reset_step_button_p1.pack(side=tk.LEFT)

        # Matrix Input
        ttk.Label(input_frame, text="Enter Coefficients [A|b]:", style='Title.TLabel').pack(fill='x', pady=(10, 5))
        self.matrix_input_container = ttk.Frame(input_frame)
        self.matrix_input_container.pack(fill='x', expand=False, pady=(0, 15))
        self.generate_matrix_input()

        # Method Selection
        ttk.Label(input_frame, text="Solving Method:", style='Title.TLabel').pack(fill='x', pady=(10, 5))
        self.method_options = ["Gauss Elimination", "Gauss-Jordan", "LU Decomposition", "Jacobi-Iteration",
                               "Gauss-Seidel"]
        self.method_dropdown = ttk.Combobox(input_frame, textvariable=self.method_var, values=self.method_options,
                                            state="readonly", font=('Arial', 10))
        self.method_dropdown.pack(fill='x', pady=(0, 15))
        self.method_var.trace_add("write", self.update_parameters_frame)

        # Parameters Frame
        self.params_frame = ttk.LabelFrame(input_frame, text="Method Parameters", padding="15")
        self.params_frame.pack(fill='x', pady=(10, 15))
        self.update_parameters_frame()

        # Precision
        precision_frame = ttk.Frame(input_frame)
        precision_frame.pack(fill='x', pady=(5, 10))
        ttk.Label(precision_frame, text="Precision (Significant Figures):", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Entry(precision_frame, textvariable=self.precision_var, width=10, font=('Arial', 10)).pack(side=tk.RIGHT,
                                                                                                       padx=(10, 0))

        # BONUS #3: Scaling Option
        scaling_frame = ttk.LabelFrame(input_frame, text="Advanced Options (Bonus)", padding="10")
        scaling_frame.pack(fill='x', pady=(10, 15))
        ttk.Checkbutton(scaling_frame, text="Enable Scaled Partial Pivoting",
                       variable=self.use_scaling_var,
                       command=self.update_scaling_help).pack(fill='x', padx=5, pady=5)
        self.scaling_help_label = ttk.Label(scaling_frame,
                                           text="Improves numerical stability by normalizing row magnitudes during pivot selection",
                                           font=("Arial", 9), foreground="#7F8C8D", wraplength=250)
        self.scaling_help_label.pack(fill='x', padx=5)

        # BONUS #1: Single-Step Mode for Phase 1
        single_step_frame = ttk.LabelFrame(input_frame, text="Single-Step Mode (Bonus #1)", padding="10")
        single_step_frame.pack(fill='x', pady=(10, 15))
        ttk.Checkbutton(single_step_frame, text="Enable Single-Step Mode",
                       variable=self.single_step_phase1_var).pack(fill='x', padx=5, pady=5)
        ttk.Label(single_step_frame,
                 text="View each algorithm step (pivoting, elimination, substitution, iteration)",
                 font=("Arial", 9), foreground="#7F8C8D", wraplength=250).pack(fill='x', padx=5)

        # Output Frame (Right)
        output_frame = ttk.LabelFrame(main_frame, text="Solution & Results", padding="15")
        output_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)

        ttk.Label(output_frame, text="Results Output:", style='Title.TLabel').pack(fill='x', pady=(0, 5))
        self.results_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20, width=50,
                                                      font=("Consolas", 11), state=tk.DISABLED, bg="#F0F8FF",
                                                      fg="#004D99", relief=tk.FLAT)
        self.results_text.pack(fill='both', expand=True)

        ttk.Label(output_frame, text="Details & Logs:", style='Title.TLabel').pack(fill='x', pady=(15, 5))
        self.log_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=5, width=50, font=("Consolas", 10),
                                                  state=tk.DISABLED, bg="#F5F5F5", fg="#555555", relief=tk.FLAT)
        self.log_text.pack(fill='both', expand=True)

    def update_scaling_help(self):
        """Update scaling help text based on toggle"""
        if self.use_scaling_var.get():
            self.scaling_help_label.config(
                text="✓ Enabled: Uses scaled partial pivoting for improved numerical stability"
            )
        else:
            self.scaling_help_label.config(
                text="Disabled: Uses standard partial pivoting (faster but may be less stable)"
            )

    def validate_equation(self, equation: str) -> tuple:
        """Validate equation input for root finding"""
        if not equation or equation.strip() == "":
            return False, "Equation cannot be empty"

        equation = equation.strip()

        allowed_pattern = r'^[x0-9\+\-\*/\.\(\)\s\*\*eEsincoxtaqrlgbfhp]+$'
        if not re.match(allowed_pattern, equation):
            return False, "Equation contains invalid characters. Use only: x, numbers, +, -, *, /, **, (), and math functions"

        temp_eq = equation
        for func in ['sin', 'cos', 'tan', 'exp', 'sqrt', 'log', 'abs']:
            temp_eq = temp_eq.replace(func, '')

        invalid_vars = re.findall(r'[a-wyz]', temp_eq, re.IGNORECASE)
        if invalid_vars:
            return False, f"Invalid variable(s): {', '.join(set(invalid_vars))}. Only 'x' is allowed"

        if 'x' not in equation:
            return False, "Equation must contain variable 'x'"

        if equation.count('(') != equation.count(')'):
            return False, "Mismatched parentheses"

        invalid_patterns = [
            (r'\*{3,}', "Invalid operator: too many consecutive '*'"),
            (r'/{2,}', "Invalid operator: consecutive '/'"),
            (r'\+{2,}', "Invalid operator: consecutive '+'"),
            (r'\d+\.\d+\.\d+', "Invalid number format: multiple decimal points")
        ]

        for pattern, error_msg in invalid_patterns:
            if re.search(pattern, equation):
                return False, error_msg

        try:
            safe_expr = equation.replace('^', '**')
            safe_expr = safe_expr.replace('sin', 'math.sin')
            safe_expr = safe_expr.replace('cos', 'math.cos')
            safe_expr = safe_expr.replace('tan', 'math.tan')
            safe_expr = safe_expr.replace('exp', 'math.exp')
            safe_expr = safe_expr.replace('sqrt', 'math.sqrt')
            safe_expr = safe_expr.replace('log', 'math.log')
            safe_expr = safe_expr.replace('abs', 'math.fabs')

            test_result = eval(safe_expr, {"x": 1.0, "math": math, "__builtins__": {}})

            if not isinstance(test_result, (int, float)):
                return False, "Equation must evaluate to a number"

        except SyntaxError as e:
            return False, f"Syntax error in equation: {str(e)}"
        except Exception as e:
            return False, f"Invalid equation: {str(e)}"

        return True, "Equation is valid"

    def setup_phase2_ui(self, parent):
        """Setup Phase 2 Root Finding UI"""
        main_frame = ttk.Frame(parent, padding="20 20 20 20")
        main_frame.pack(fill='both', expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2, minsize=400)
        main_frame.columnconfigure(1, weight=3, minsize=600)

        # Input Frame (Left)
        scroll_frame = ScrollableFrame(main_frame)
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        scroll_frame.scrollable_frame.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(scroll_frame.scrollable_frame, text="Equation & Method Selection", padding="15")
        input_frame.pack(fill='both', expand=True)
        input_frame.columnconfigure(0, weight=1)

        # Equation Input
        ttk.Label(input_frame, text="Equation (use x as variable):", style='Title.TLabel').pack(fill='x', pady=(10, 5))
        ttk.Label(input_frame, text="Examples: x**2 - 4, sin(x) - x/2, exp(x) - 3*x", font=("Arial", 9),
                  foreground="#7F8C8D").pack(fill='x', pady=(0, 5))
        self.equation_entry = ttk.Entry(input_frame, textvariable=self.equation_var, font=('Arial', 10))
        self.equation_entry.pack(fill='x', pady=(0, 15))

        # Plotting Button
        plot_button_frame = ttk.Frame(input_frame)
        plot_button_frame.pack(fill='x', pady=(5, 15))
        ttk.Button(plot_button_frame, text="PLOT FUNCTION",
                   command=self.plot_function).pack(fill='x')
        ttk.Label(plot_button_frame,
                  text="Plot to help choose initial guess(es)",
                  font=("Arial", 9), foreground="#7F8C8D").pack(fill='x', pady=(5, 0))

        # Method Selection
        ttk.Label(input_frame, text="Root Finding Method:", style='Title.TLabel').pack(fill='x', pady=(10, 5))
        self.root_methods = ["Bisection", "False-Position", "Fixed Point", "Newton-Raphson", "Modified Newton-Raphson (Known m)",
                              "Modified Newton-Raphson (Unknown m)", "Secant", "Modified Secant"]
        self.root_method_dropdown = ttk.Combobox(input_frame, textvariable=self.root_method_var,
                                                 values=self.root_methods, state="readonly", font=('Arial', 10))
        self.root_method_dropdown.pack(fill='x', pady=(0, 15))
        self.root_method_var.trace_add("write", self.update_root_parameters_frame)

        # Parameters Frame
        self.root_params_frame = ttk.LabelFrame(input_frame, text="Method Parameters", padding="15")
        self.root_params_frame.pack(fill='x', pady=(10, 15))
        self.update_root_parameters_frame()

        # Precision & Stopping Criteria
        criteria_frame = ttk.LabelFrame(input_frame, text="Precision & Stopping Criteria", padding="15")
        criteria_frame.pack(fill='x', pady=(10, 15))

        ttk.Label(criteria_frame, text="Precision (Sig Figs):", style='TLabel').pack(fill='x', pady=(5, 2))
        ttk.Entry(criteria_frame, textvariable=self.root_precision_var, font=('Arial', 10)).pack(fill='x')

        ttk.Label(criteria_frame, text="Epsilon (EPS):", style='TLabel').pack(fill='x', pady=(10, 2))
        ttk.Entry(criteria_frame, textvariable=self.epsilon_var, font=('Arial', 10)).pack(fill='x')

        ttk.Label(criteria_frame, text="Max Iterations:", style='TLabel').pack(fill='x', pady=(10, 2))
        ttk.Entry(criteria_frame, textvariable=self.max_iter_root_var, font=('Arial', 10)).pack(fill='x')

        # Single Step Mode Toggle (BONUS)
        single_step_frame = ttk.Frame(input_frame)
        single_step_frame.pack(fill='x', pady=(15, 10))
        ttk.Checkbutton(single_step_frame, text="Enable Single Step Mode (Bonus #1)", variable=self.single_step_var).pack(
            side=tk.LEFT)
        ttk.Label(single_step_frame, text="View each iteration step-by-step", font=("Arial", 9),
                  foreground="#7F8C8D").pack(side=tk.LEFT, padx=(10, 0))

        # Solve & Step Buttons Frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill='x', pady=(0, 15))
        ttk.Button(button_frame, text="SOLVE EQUATION", command=self.solve_root_finding,
                   style='InlineSolve.TButton').pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        self.next_step_button = ttk.Button(button_frame, text="Next Step", command=self.next_step_root_finding,
                                           state=tk.DISABLED)
        self.next_step_button.pack(side=tk.LEFT, padx=(0, 5))
        self.reset_step_button = ttk.Button(button_frame, text="Reset", command=self.reset_step_mode, state=tk.DISABLED)
        self.reset_step_button.pack(side=tk.LEFT)

        # Output Frame (Right)
        output_scroll_frame = ScrollableFrame(main_frame)
        output_scroll_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        output_container = output_scroll_frame.scrollable_frame
        output_container.columnconfigure(0, weight=1)

        output_frame = ttk.LabelFrame(output_container, text="Root & Results", padding="15")
        output_frame.pack(fill='both', expand=True)
        output_frame.columnconfigure(0, weight=1)

        # Plot Display
        ttk.Label(output_frame, text="Function Plot:", style='Title.TLabel').pack(fill='x', pady=(0, 5))

        self.plot_frame = ttk.Frame(output_frame, relief=tk.SUNKEN, borderwidth=1)
        self.plot_frame.pack(fill='both', expand=True, pady=(0, 10))

        self.plot_placeholder = ttk.Label(
            self.plot_frame,
            text="Click 'PLOT FUNCTION' to visualize\nand choose initial guesses",
            font=("Arial", 10),
            foreground="#95A5A6",
            justify=tk.CENTER)
        self.plot_placeholder.pack(expand=True)

        # Results Output
        ttk.Label(output_frame, text="Results Output:", style='Title.TLabel').pack(fill='x', pady=(15, 5))

        self.root_results_text = scrolledtext.ScrolledText(
            output_frame, wrap=tk.WORD, height=15, width=50,
            font=("Consolas", 10), state=tk.DISABLED,
            bg="#F0F8FF", fg="#004D99", relief=tk.FLAT)
        self.root_results_text.pack(fill='both', expand=True, pady=(0, 10))

        # Iteration Details
        ttk.Label(output_frame, text="Iteration Details:", style='Title.TLabel').pack(fill='x', pady=(10, 5))

        self.root_log_text = scrolledtext.ScrolledText(
            output_frame, wrap=tk.WORD, height=12, width=50,
            font=("Consolas", 9), state=tk.DISABLED,
            bg="#F5F5F5", fg="#555555", relief=tk.FLAT)
        self.root_log_text.pack(fill='both', expand=True)

    def clear_params_frame(self):
        """Clear Phase 1 parameters frame"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

    def clear_root_params_frame(self):
        """Clear Phase 2 parameters frame"""
        for widget in self.root_params_frame.winfo_children():
            widget.destroy()

    def generate_matrix_input(self):
        """Generate N×(N+1) matrix input grid"""
        try:
            N = int(self.n_var.get())
            if N <= 0:
                messagebox.showwarning("Input Warning", "N must be between 1 and 10")
                self.n_var.set("3")
                N = 3
        except ValueError:
            messagebox.showerror("Input Error", "N must be an integer")
            self.n_var.set("3")
            return

        for widget in self.matrix_input_container.winfo_children():
            widget.destroy()

        self.matrix_entry_widgets = []

        for j in range(N):
            ttk.Label(self.matrix_input_container, text=f"X{j + 1}", font=("Arial", 10, "bold"),
                      foreground="#2980B9").grid(row=0, column=j, padx=4, pady=4)
        ttk.Label(self.matrix_input_container, text=" | B", font=("Arial", 10, "bold"), foreground="#E74C3C").grid(
            row=0, column=N, padx=8, pady=4)

        for i in range(N):
            row_entries = []
            for j in range(N + 1):
                entry = ttk.Entry(self.matrix_input_container, width=5, font=('Consolas', 10))
                if j == N:
                    entry.grid(row=i + 1, column=j, padx=(10, 2), pady=2, sticky='ew')
                    entry.config(foreground="#E74C3C")
                else:
                    entry.grid(row=i + 1, column=j, padx=2, pady=2, sticky='ew')
                    entry.config(foreground="#2980B9")
                row_entries.append(entry)
            self.matrix_entry_widgets.append(row_entries)

        initial_data = [[4.0, 1.0, -1.0, 3.0], [2.0, 7.0, 1.0, 19.0], [1.0, -3.0, 12.0, 31.0]]
        for i in range(min(N, 3)):
            for j in range(N + 1):
                if i < len(initial_data) and j < len(initial_data[i]):
                    self.matrix_entry_widgets[i][j].delete(0, tk.END)
                    self.matrix_entry_widgets[i][j].insert(0, str(initial_data[i][j]))

    def update_parameters_frame(self, *args):
        """Update Phase 1 parameters based on selected method"""
        self.clear_params_frame()
        method = self.method_var.get()

        if method == "LU Decomposition":
            ttk.Label(self.params_frame, text="LU Form:", style='TLabel').pack(fill='x', pady=(5, 5))
            ttk.Combobox(self.params_frame, textvariable=self.lu_form_var, values=["Doolittle", "Crout", "Cholesky"],
                         state="readonly", font=('Arial', 10)).pack(fill='x', pady=(0, 10))
        elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            ttk.Label(self.params_frame, text="Initial Guess (comma-separated):", style='TLabel').pack(fill='x',
                                                                                                       pady=(5, 5))
            ttk.Entry(self.params_frame, textvariable=self.initial_guess_var, font=('Arial', 10)).pack(fill='x',
                                                                                                       pady=(0, 10))
            ttk.Label(self.params_frame, text="Max Iterations:", style='TLabel').pack(fill='x', pady=(5, 2))
            ttk.Entry(self.params_frame, textvariable=self.max_iter_var, font=('Arial', 10)).pack(fill='x')
            ttk.Label(self.params_frame, text="Error Tolerance:", style='TLabel').pack(fill='x', pady=(10, 2))
            ttk.Entry(self.params_frame, textvariable=self.error_tol_var, font=('Arial', 10)).pack(fill='x')

    def update_root_parameters_frame(self, *args):
        """Update Phase 2 parameters based on selected method"""
        self.clear_root_params_frame()
        method = self.root_method_var.get()

        if method == "Bisection":
            ttk.Label(self.root_params_frame, text="Interval [a, b]:", style='TLabel').pack(fill='x', pady=(5, 5))
            a_frame = ttk.Frame(self.root_params_frame)
            a_frame.pack(fill='x', pady=(0, 5))
            ttk.Label(a_frame, text="a:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(a_frame, textvariable=self.interval_a_var, width=10, font=('Arial', 10)).pack(side=tk.LEFT)
            b_frame = ttk.Frame(self.root_params_frame)
            b_frame.pack(fill='x')
            ttk.Label(b_frame, text="b:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(b_frame, textvariable=self.interval_b_var, width=10, font=('Arial', 10)).pack(side=tk.LEFT)

        elif method == "Fixed Point":
            ttk.Label(self.root_params_frame, text="Initial Guess (x₀):", style='TLabel').pack(fill='x', pady=(5, 5))
            ttk.Label(self.root_params_frame, text="Equation g(x) should be in form: x_new = g(x)", font=("Arial", 9),
                      foreground="#7F8C8D").pack(fill='x', pady=(0, 5))
            ttk.Entry(self.root_params_frame, textvariable=self.initial_guess_root_var, font=('Arial', 10)).pack(
                fill='x')

        elif method == "False-Position":
            ttk.Label(self.root_params_frame, text="Interval [a, b]:", style='TLabel').pack(fill='x', pady=(5, 5))
            a_frame = ttk.Frame(self.root_params_frame)
            a_frame.pack(fill='x', pady=(0, 5))
            ttk.Label(a_frame, text="a:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(a_frame, textvariable=self.interval_a_var, width=10, font=('Arial', 10)).pack(side=tk.LEFT)
            b_frame = ttk.Frame(self.root_params_frame)
            b_frame.pack(fill='x')
            ttk.Label(b_frame, text="b:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(b_frame, textvariable=self.interval_b_var, width=10, font=('Arial', 10)).pack(side=tk.LEFT)

        elif method == "Newton-Raphson":
            ttk.Label(self.root_params_frame, text="Initial Guess (x₀):", style='TLabel').pack(fill='x', pady=(5, 5))
            ttk.Entry(self.root_params_frame, textvariable=self.initial_guess_root_var, font=('Arial', 10)).pack(
                fill='x')

        elif method == "Modified Newton-Raphson (Known m)":
            ttk.Label(self.root_params_frame, text="Initial Guess (x₀):", style='TLabel').pack(fill='x', pady=(5, 5))
            ttk.Entry(self.root_params_frame, textvariable=self.initial_guess_root_var, font=('Arial', 10)).pack(fill='x')

            ttk.Label(self.root_params_frame, text="Multiplicity (m):", style='TLabel').pack(fill='x', pady=(5, 5))
            self.m_var = tk.StringVar(self.master, value="1")
            ttk.Entry(self.root_params_frame, textvariable=self.m_var, font=('Arial', 10)).pack(fill='x')

        elif method == "Modified Newton-Raphson (Unknown m)":
            ttk.Label(self.root_params_frame, text="Initial Guess (x₀):", style='TLabel').pack(fill='x', pady=(5, 5))
            ttk.Entry(self.root_params_frame, textvariable=self.initial_guess_root_var, font=('Arial', 10)).pack(fill='x')

        elif method == "Secant":
            initial_guess_frame = ttk.Frame(self.root_params_frame)
            initial_guess_frame.pack(fill='x', pady=(0, 5))
            ttk.Label(initial_guess_frame, text="x₋₁:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(initial_guess_frame, textvariable=self.initial_guess_root_var, width=10, font=('Arial', 10)).pack(side=tk.LEFT)

            second_guess_frame = ttk.Frame(self.root_params_frame)
            second_guess_frame.pack(fill='x', pady=(0, 5))
            ttk.Label(second_guess_frame, text="x₀  :").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(second_guess_frame, textvariable=self.second_guess_root_var, width=10, font=('Arial', 10)).pack(
                side=tk.LEFT)

        elif method == "Modified Secant":
            initial_guess_frame = ttk.Frame(self.root_params_frame)
            initial_guess_frame.pack(fill='x', pady=(0, 5))
            ttk.Label(initial_guess_frame, text="x₀:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(initial_guess_frame, textvariable=self.initial_guess_root_var, width=10, font=('Arial', 10)).pack(
                side=tk.LEFT)

            delta_frame = ttk.Frame(self.root_params_frame)
            delta_frame.pack(fill='x', pady=(0, 5))
            ttk.Label(delta_frame, text="ẟ  :").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Entry(delta_frame, textvariable=self.delta_var, width=10, font=('Arial', 10)).pack(side=tk.LEFT)

    def solve_linear_system(self):
        """Solve Phase 1 linear system with optional single-step and scaling bonuses"""
        try:
            N = int(self.n_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "N must be an integer")
            return

        try:
            precision = int(self.precision_var.get() or 5)
            if precision <= 0 or precision > 15:
                raise ValueError("Precision must be 1-15")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        parsed_matrix = self.solver.parse_input(self.matrix_entry_widgets, N)
        if parsed_matrix is None:
            return

        A, b = parsed_matrix
        method = self.method_var.get()
        params = {}

        if method == "LU Decomposition":
            params["LU Form"] = self.lu_form_var.get()
        elif method in ["Jacobi-Iteration", "Gauss-Seidel"]:
            try:
                guess_list = [float(x.strip()) for x in self.initial_guess_var.get().split(',')]
                if len(guess_list) != N:
                    raise ValueError(f"Initial guess must have {N} components")
                params["Initial Guess"] = guess_list
                params["max_iter_var"] = int(self.max_iter_var.get())
                params["error_tol_var"] = float(self.error_tol_var.get())
            except ValueError as e:
                messagebox.showerror("Input Error", str(e))
                return

        # Create SystemData with scaling and single-step support
        system_data = SystemData(copy.deepcopy(A), copy.deepcopy(b), method, precision, params,
                                use_scaling=self.use_scaling_var.get())

        # Track execution time
        start_time = time.time()
        results = self.solver.solve(system_data)
        self.execution_time_phase1 = time.time() - start_time

        if results["success"]:
            # Store steps for single-step mode
            self.all_steps_phase1 = results.get("steps", [])

            # Check if single-step mode is enabled
            if self.single_step_phase1_var.get():
                self.step_mode_active_phase1 = True
                self.current_step_index_phase1 = 0
                self.next_step_button_p1.config(state=tk.NORMAL)
                self.reset_step_button_p1.config(state=tk.NORMAL)
                self.display_step_phase1()
            else:
                # Display full solution
                sol_text = ""
                for i, val in enumerate(results["sol"]):
                    formatted_val = f"{val:.{precision}f}".rstrip('0').rstrip('.')
                    sol_text += f"X{i + 1} = {formatted_val}\n"
                output_text = f"--- Solution ---\n\n{sol_text}"
                scaling_note = "\n[Using Scaled Partial Pivoting]" if self.use_scaling_var.get() else ""
                log_text = f"Method: {results['method_used']}{scaling_note}\nPrecision: {results['precision']} sig figs\nExecution Time: {self.execution_time_phase1:.6f}s\n"

                self.update_results_display(output_text, log_text, phase=1)
        else:
            output_text = f"ERROR:\n{results.get('error_message', 'Unknown error')}"
            log_text = f"Execution Time: {self.execution_time_phase1:.6f}s\n"
            self.update_results_display(output_text, log_text, phase=1)

    def display_step_phase1(self):
        """Display current step in Phase 1 single-step mode"""
        if self.current_step_index_phase1 >= len(self.all_steps_phase1):
            self.show_final_summary_phase1()
            return

        current_step = self.all_steps_phase1[self.current_step_index_phase1]
        progress = f"Step {self.current_step_index_phase1 + 1} of {len(self.all_steps_phase1)}"

        output_text = f"--- SINGLE STEP MODE (Phase 1) ---\n\n"
        output_text += f"{progress}\n"
        output_text += f"Method: {self.method_var.get()}\n"
        output_text += f"Precision: {self.precision_var.get()} sig figs\n\n"
        output_text += "Current Step Details:\n"
        output_text += "-" * 60 + "\n"

        # Format step information
        if isinstance(current_step, dict):
            for key, value in current_step.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                    output_text += f"{key}: [Matrix Data]\n"
                elif isinstance(value, (int, float)):
                    output_text += f"{key:.<40} {value:.6f}\n"
                else:
                    output_text += f"{key:.<40} {value}\n"
        else:
            output_text += str(current_step) + "\n"

        output_text += "-" * 60 + "\n\n"
        output_text += "Press 'Next Step' to continue or 'Reset' to start over"

        log_text = f"Execution Time: {self.execution_time_phase1:.6f}s\n\n"
        log_text += f"Steps Completed: {self.current_step_index_phase1 + 1}/{len(self.all_steps_phase1)}\n\n"
        log_text += "Step History:\n"
        for i, step in enumerate(self.all_steps_phase1[:self.current_step_index_phase1 + 1]):
            if isinstance(step, dict) and "operation" in step:
                log_text += f"[{i + 1}] {step.get('operation', 'Unknown')}\n"
            else:
                log_text += f"[{i + 1}] {str(step)[:50]}...\n"

        self.update_results_display(output_text, log_text, phase=1)

    def next_step_phase1(self):
        """Move to next step in Phase 1 single-step mode"""
        if self.step_mode_active_phase1:
            self.current_step_index_phase1 += 1
            self.display_step_phase1()

    def reset_step_phase1(self):
        """Reset Phase 1 step mode and go back to beginning"""
        self.step_mode_active_phase1 = False
        self.current_step_index_phase1 = 0
        self.next_step_button_p1.config(state=tk.DISABLED)
        self.reset_step_button_p1.config(state=tk.DISABLED)

        # Display full solution
        if self.all_steps_phase1:
            sol_text = "Solution from Last Solve:\n"
            if isinstance(self.all_steps_phase1[-1], dict) and "solution" in self.all_steps_phase1[-1]:
                solution = self.all_steps_phase1[-1]["solution"]
                precision = int(self.precision_var.get() or 5)
                for i, val in enumerate(solution):
                    formatted_val = f"{val:.{precision}f}".rstrip('0').rstrip('.')
                    sol_text += f"X{i + 1} = {formatted_val}\n"

            output_text = f"--- Solution ---\n\n{sol_text}"
            scaling_note = "\n[Using Scaled Partial Pivoting]" if self.use_scaling_var.get() else ""
            log_text = f"Method: {self.method_var.get()}{scaling_note}\nPrecision: {self.precision_var.get()} sig figs\nExecution Time: {self.execution_time_phase1:.6f}s\n"

            self.update_results_display(output_text, log_text, phase=1)

    def show_final_summary_phase1(self):
        """Show final summary after all steps completed in Phase 1"""
        output_text = f"--- ALL STEPS COMPLETED ---\n\n"
        output_text += f"Method: {self.method_var.get()}\n"
        output_text += f"Total Steps: {len(self.all_steps_phase1)}\n\n"

        # Extract final solution if available
        if self.all_steps_phase1 and isinstance(self.all_steps_phase1[-1], dict):
            if "solution" in self.all_steps_phase1[-1]:
                solution = self.all_steps_phase1[-1]["solution"]
                precision = int(self.precision_var.get() or 5)
                output_text += "✓ Solution Found:\n"
                for i, val in enumerate(solution):
                    formatted_val = f"{val:.{precision}f}".rstrip('0').rstrip('.')
                    output_text += f"  X{i + 1} = {formatted_val}\n"

        output_text += f"\n✓ Execution Time: {self.execution_time_phase1:.6f}s\n"
        output_text += "All steps completed. Press 'Reset' to solve again."

        log_text = "Complete Step History:\n\n"
        for i, step in enumerate(self.all_steps_phase1):
            if isinstance(step, dict) and "operation" in step:
                log_text += f"[{i + 1}] {step.get('operation', 'Unknown')}\n"
                if "description" in step:
                    log_text += f"     {step['description']}\n"
            else:
                log_text += f"[{i + 1}] {str(step)[:60]}...\n"

        self.update_results_display(output_text, log_text, phase=1)
        self.next_step_button_p1.config(state=tk.DISABLED)

    def solve_root_finding(self):
        """Solve Phase 2 root finding problem"""
        equation = self.equation_var.get().strip()

        is_valid, error_msg = self.validate_equation(equation)
        if not is_valid:
            messagebox.showerror("Invalid Equation", error_msg)
            return

        try:
            precision = int(self.root_precision_var.get() or 5)
            eps = float(self.epsilon_var.get() or 0.00001)
            max_iter = int(self.max_iter_root_var.get() or 50)
            if precision <= 0 or precision > 15:
                raise ValueError("Precision must be 1-15")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")
            return

        method = self.root_method_var.get()
        params = {
            "epsilon": eps,
            "max_iterations": max_iter
        }

        try:
            if method == "Bisection":
                params["interval_a"] = float(self.interval_a_var.get())
                params["interval_b"] = float(self.interval_b_var.get())
            elif method == "Fixed Point":
                params["initial_guess"] = float(self.initial_guess_root_var.get())
            elif method == "False-Position":
                params["interval_a"] = float(self.interval_a_var.get())
                params["interval_b"] = float(self.interval_b_var.get())
            elif method == "Newton-Raphson":
                params["initial_guess"] = float(self.initial_guess_root_var.get())
            elif method == "Modified Newton-Raphson (Known m)":
                params["initial_guess"] = float(self.initial_guess_root_var.get())
                try:
                    params["multiplicity"] = float(self.m_var.get())
                except Exception as e:
                    messagebox.showerror("Input Error", "Please enter a valid multiplicity 'm'")
                    return
            elif method == "Modified Newton-Raphson (Unknown m)":
                params["initial_guess"] = float(self.initial_guess_root_var.get())
            elif method == "Secant":
                params["initial_guess"] = float(self.initial_guess_root_var.get())
                params["second_guess"] = float(self.second_guess_root_var.get())
            elif method == "Modified Secant":
                params["initial_guess"] = float(self.initial_guess_root_var.get())
                params["delta"] = float(self.delta_var.get())

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        start_time = time.time()
        try:
            root_data = RootFinderData(equation, method, precision, params)
            solver = RootFinderFactory.get_solver(root_data)
            results = solver.solve()
            self.execution_time = time.time() - start_time

            self.all_steps = results.get("steps", [])
            self.current_root = results["root"]
            self.current_iterations = results["iterations"]
            self.current_rel_error = results["rel_error"]
            self.current_correct_sigs = results["correct_sig_figs"]
            self.current_method = method
            self.current_equation = equation

            if self.single_step_var.get():
                self.step_mode_active = True
                self.current_step_index = 0
                self.next_step_button.config(state=tk.NORMAL)
                self.reset_step_button.config(state=tk.NORMAL)
                self.display_step_mode()
            else:
                self.display_full_solution()

        except Exception as e:
            output_text = f"ERROR:\n{str(e)}"
            log_text = f"Execution Time: {time.time() - start_time:.6f}s"
            self.update_results_display(output_text, log_text, phase=2)

    def display_full_solution(self):
        """Display complete solution (non-step mode)"""
        output_text = f"--- Root Finding Results ---\n\n"
        output_text += f"Method: {self.current_method}\n"
        output_text += f"Equation: {self.current_equation}\n\n"
        output_text += f"Approximate Root: {self.current_root}\n"
        output_text += f"Iterations: {self.current_iterations}\n"
        output_text += f"Relative Error: {self.current_rel_error}\n"
        output_text += f"Correct Sig Figs: {self.current_correct_sigs}\n"
        output_text += f"Execution Time: {self.execution_time:.6f}s\n"

        steps_text = "All Iterations:\n\n"
        for step in self.all_steps:
            steps_text += str(step) + "\n"

        self.update_results_display(output_text, steps_text, phase=2)

    def display_step_mode(self):
        """Display current step in step-by-step mode"""
        if self.current_step_index >= len(self.all_steps):
            self.show_final_summary()
            return

        current_step = self.all_steps[self.current_step_index]
        progress = f"Step {self.current_step_index + 1} of {len(self.all_steps)}"

        output_text = f"--- SINGLE STEP MODE ---\n\n"
        output_text += f"{progress}\n"
        output_text += f"Method: {self.current_method}\n"
        output_text += f"Equation: {self.current_equation}\n\n"
        output_text += "Current Iteration Data:\n"
        output_text += "-" * 50 + "\n"

        for key, value in current_step.items():
            output_text += f"{key:.<30} {value}\n"

        output_text += "-" * 50 + "\n\n"
        output_text += "Press 'Next Step' to continue or 'Reset' to start over"

        steps_text = f"Completed: {self.current_step_index + 1}/{len(self.all_steps)} steps\n\n"
        steps_text += "Iteration History:\n"
        for i, step in enumerate(self.all_steps[:self.current_step_index + 1]):
            steps_text += f"Step {i + 1}: {step}\n"

        self.update_results_display(output_text, steps_text, phase=2)

    def next_step_root_finding(self):
        """Move to next step in step-by-step mode"""
        if self.step_mode_active:
            self.current_step_index += 1
            self.display_step_mode()

    def reset_step_mode(self):
        """Reset step mode and go back to beginning"""
        self.step_mode_active = False
        self.current_step_index = 0
        self.next_step_button.config(state=tk.DISABLED)
        self.reset_step_button.config(state=tk.DISABLED)
        self.display_full_solution()

    def show_final_summary(self):
        """Show final summary after all steps completed"""
        output_text = f"--- ROOT FINDING COMPLETE ---\n\n"
        output_text += f"Method: {self.current_method}\n"
        output_text += f"Equation: {self.current_equation}\n\n"
        output_text += f"✓ Approximate Root: {self.current_root}\n"
        output_text += f"✓ Total Iterations: {self.current_iterations}\n"
        output_text += f"✓ Relative Error: {self.current_rel_error}\n"
        output_text += f"✓ Correct Sig Figs: {self.current_correct_sigs}\n"
        output_text += f"✓ Execution Time: {self.execution_time:.6f}s\n\n"
        output_text += "All steps completed. Press 'Reset' to start over."

        steps_text = "Final Iteration Summary:\n\n"
        for i, step in enumerate(self.all_steps):
            steps_text += f"Step {i + 1}: {step}\n"

        self.update_results_display(output_text, steps_text, phase=2)
        self.next_step_button.config(state=tk.DISABLED)

    def update_results_display(self, text: str, log: str, phase: int = 1):
        """Update results display for Phase 1 or Phase 2"""
        if phase == 1:
            for widget in [self.results_text, self.log_text]:
                widget.config(state=tk.NORMAL)
                widget.delete(1.0, tk.END)
            self.results_text.insert(tk.END, text)
            self.log_text.insert(tk.END, log)
            for widget in [self.results_text, self.log_text]:
                widget.config(state=tk.DISABLED)
        else:
            for widget in [self.root_results_text, self.root_log_text]:
                widget.config(state=tk.NORMAL)
                widget.delete(1.0, tk.END)
            self.root_results_text.insert(tk.END, text)
            self.root_log_text.insert(tk.END, log)
            for widget in [self.root_results_text, self.root_log_text]:
                widget.config(state=tk.DISABLED)

    def plot_function(self):
        """Plot function to help choose initial guesses"""
        equation = self.equation_var.get().strip()

        is_valid, error_msg = self.validate_equation(equation)
        if not is_valid:
            messagebox.showerror("Invalid Equation", error_msg)
            return

        method = self.root_method_var.get()

        try:
            try:
                a = float(self.interval_a_var.get() or -10)
                b = float(self.interval_b_var.get() or 10)
                if a >= b:
                    a, b = -10, 10
            except ValueError:
                a, b = -10, 10

            temp_params = {"epsilon": 0.00001, "max_iterations": 1}
            root_data = RootFinderData(equation, "Bisection", 5, temp_params)
            solver = RootFinderFactory.get_solver(root_data)

            x_vals = np.linspace(a, b, 500)
            y_vals = []

            for x in x_vals:
                try:
                    y = solver.evaluate(x)
                    y_vals.append(y if abs(y) < 1e10 else None)
                except:
                    y_vals.append(None)

            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)

            if method == "Fixed Point":
                ax.plot(x_vals, y_vals, 'b-', label='g(x)', linewidth=2)
                ax.plot(x_vals, x_vals, 'r--', label='y = x', linewidth=2)
                ax.set_title(f'Fixed Point: g(x)=0 ∩ y=x\n{equation}', fontsize=10)
            else:
                ax.plot(x_vals, y_vals, 'b-', label='f(x)', linewidth=2)
                ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='y = 0')
                ax.set_title(f'{method}: f(x) = 0\n{equation}', fontsize=10)

            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x', fontsize=10, fontweight='bold')
            ax.set_ylabel('y', fontsize=10, fontweight='bold')
            ax.legend(loc='best')
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            messagebox.showinfo("Plot Success", f"Plotted from x={a} to x={b}")

        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not plot:\n{str(e)}")
            self.plot_placeholder.pack(expand=True)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("Starting Numerical Solver - Phase 1 & 2...")
    root = tk.Tk()
    app = NumericalSolverGUI(root)
    root.mainloop()