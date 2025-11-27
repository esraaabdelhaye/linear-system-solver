import math
import numpy as np
import sympy as sp

class AbstractSolver2:
    def __init__(self, A, b, precision=6, single_step=False, symbolic_mode=False):
       
        self.user_A = A
        self.user_b = b
        self.precision = precision
        self.single_step = single_step
        self.symbolic_mode = symbolic_mode
        self.steps = []

        # Detect symbolic
        self._detect_symbolic_or_numeric()



    def _detect_symbolic_or_numeric(self):
        contains_symbol = False

        # Check elements for symbolic values
        for row in self.user_A:
            for val in row:
                if isinstance(val, str) or isinstance(val, sp.Basic):       # 'a', 'b', 'c',    sympy symbol
                    contains_symbol = True
                

        for val in self.user_b:
            if isinstance(val, str) or isinstance(val, sp.Basic):
                contains_symbol = True

        # CASE 1 → Matrix has symbols → use SymPy always
        if contains_symbol:
            self.symbolic = True
            self.A = sp.Matrix(self._to_symbolic(self.user_A))
            self.b = sp.Matrix(self._to_symbolic(self.user_b))
        else:
            # CASE 2 → Numeric + symbolic_mode = True → force Symbolic
            if self.symbolic_mode:
                self.symbolic = True
                self.A = sp.Matrix(self._to_symbolic(self.user_A))
                self.b = sp.Matrix(self._to_symbolic(self.user_b))
            else:
                # CASE 3 → Numeric normal case
                self.symbolic = False
                self.A = np.array(self.user_A, dtype=float)
                self.b = np.array(self.user_b, dtype=float)

        self.n = len(self.A)

    # Convert numeric matrix to symbolic a11, a12...
    def _to_symbolic(self, A):
        symbolic_A = []
        for i, row in enumerate(A):
            symbolic_row = []
            for j, val in enumerate(row):
                if isinstance(val, (int, float)):
                    symbolic_row.append(sp.Symbol(f"a{i+1}{j+1}"))
                else:
                    symbolic_row.append(sp.sympify(val))
            symbolic_A.append(symbolic_row)
        return symbolic_A



    # VALIDATION
    def validate(self):
        if self.symbolic:
            if self.A.rows != self.A.cols:
                raise ValueError("Matrix must be square")
            if self.A.rows != len(self.b):
                raise ValueError("b's size must equal A's size")
        else:
            # NumPy
            if any(len(row) != self.n for row in self.A):
                raise ValueError("Matrix must be square")
            if len(self.A) != len(self.b):
                raise ValueError("b's size must equal A's size")



    def solve(self):
        raise NotImplementedError("Subclasses must implement solve()")



    # ROUNDING FOR NUMERIC MODE ONLY
    def round_sig_fig(self, x, n=None):
        if self.symbolic:
            return x  # do nothing for symbolic
        if n is None:
            n = self.precision
        if x == 0:
            return 0.0
        return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)



    # SCALING FOR NUMERIC MODE ONLY
    def get_scales(self):
        if self.symbolic:
            return None
        scales = []
        for i in range(self.n):
            max_val = max(abs(val) for val in self.A[i])
            scales.append(max_val if max_val != 0 else 1)
        return scales


    def add_step(self, step):
        if self.single_step:
            self.steps.append(step)
