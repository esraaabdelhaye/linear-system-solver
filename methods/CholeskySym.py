
import sympy as sp
import time
from AbstractSolver2 import AbstractSolver2

class Cholesky(AbstractSolver2):
    def __init__(self, A, b, precision=6, single_step=False):     # Convert inputs to SymPy matrices
        self.A = sp.Matrix(A)  
        self.b = sp.Matrix(b) 
        self.n = self.A.shape[0]  # number of variables
        self.precision = precision
        self.single_step = single_step

        # Initialize placeholders
        self.L = None
        self.y = None
        self.x = None
        self.steps = []  # stores steps if single_step=True

    def step_mode(self, description, L=None, y=None, x=None, row_steps=None):
            if self.single_step:
                self.recorder.add_step(description, L=L, y=y, x=x, row_steps=row_steps)



    def check_valid(self):
        # Check if symmetric
        if self.A != self.A.T:
            raise ValueError("Matrix must be symmetric")

        # Get approximate eigenvalues
        eigenvalues = [complex(ev.evalf()) for ev in self.A.eigenvals()]         # ev.evalf() converts each eigenvalue from symbolic to numeric approximation.

        # Reject if any eigenvalue has a significant imaginary part
        for ev in eigenvalues:
            if abs(ev.imag) > 1e-10:  
                raise ValueError("Matrix has complex eigenvalues — invalid for Cholesky")

        # Check positive-definite using real part
        if any(ev.real <= 0 for ev in eigenvalues):
            raise ValueError("Matrix is not positive-definite")

        return True
    


    def factorize(self):  # Compute L
        self.check_valid()  # Ensure matrix is valid before factorization
        n = self.n
        A = self.A
        L = sp.zeros(n)
        prec = self.precision

        for k in range(n):
            # Sum of squares for diagonal element
            sum_diag = sp.N(0, prec)
            for j in range(k):
                sum_diag = sp.N(sum_diag + sp.N(L[k, j] * L[k, j], prec), prec)
            # Diagonal element
            diff = sp.N(A[k, k] - sum_diag, prec)
            L[k, k] = sp.N(sp.sqrt(diff), prec)
            # Store step for diagonal element
            self.step_mode(f'L[{k},{k}] computed: sqrt({A[k,k]} - {sum_diag}) = {L[k,k]}', L=L)
            # self.step_mode("L[0,0] computed", L=self.L)


            # Column below diagonal
            for i in range(k+1, n):
                sum_off = sp.N(0, prec)
                for j in range(k):
                    sum_off = sp.N(sum_off + sp.N(L[i, j] * L[k, j], prec), prec)
                numerator = sp.N(A[i, k] - sum_off, prec)
                L[i, k] = sp.N(numerator / L[k, k], prec)
                # Store step for off-diagonal element
                self.step_mode(f'L[{i},{k}] computed: ({A[i,k]} - {sum_off}) / {L[k,k]} = {L[i,k]}', L=L)
                # self.step_mode("L[0,0] computed", L=self.L)


        self.L = L
        return L
    


    def forward_substitute(self):   # Solve Ly = b
        if self.L is None:
            raise ValueError("L is not computed")

        n = self.n
        y = sp.zeros(n, 1)
        prec = self.precision  

        for i in range(n):
            sum_Ly = sp.N(0, prec)
            # Compute sum of L[i,j] * y[j] for previous j
            for j in range(i):
                sum_Ly = sp.N(sum_Ly + sp.N(self.L[i, j] * y[j], prec), prec)

            # Solve for current y[i]
            numerator = sp.N(self.b[i] - sum_Ly, prec)
            y[i] = sp.N(numerator / self.L[i, i], prec)
            # Store step with formula and computed value
            self.step_mode(f'y[{i}] computed: ({self.b[i]} - {sum_Ly}) / {self.L[i,i]} = {y[i]}', L=self.L, y=y)

        self.y = y  # Save forward substitution result
        return y



    def backward_substitute(self):      # Solve Lᵀ x = y 
        if self.L is None or self.y is None:
            raise ValueError("L or Forward substitution is not computed")

        n = self.n
        x = sp.zeros(n, 1)
        prec = self.precision

        for i in reversed(range(n)):
            sum_LTx = sp.N(0, prec)
            # Compute sum of L[j,i] * x[j] for later j
            for j in range(i+1, n):
                sum_LTx = sp.N(sum_LTx + sp.N(self.L[j, i] * x[j], prec), prec)

            # Solve for current x[i]
            numerator = sp.N(self.y[i] - sum_LTx, prec)
            x[i] = sp.N(numerator / self.L[i, i], prec)
            # Store step with formula and computed value
            self.step_mode(f'x[{i}] computed: ({self.y[i]} - {sum_LTx}) / {self.L[i,i]} = {x[i]}', L=self.L, y=self.y, x=x)

        self.x = x  # Save backward substitution result
        return self.x


    def solve(self):        # Full solve
    
        start_time = time.time()      
        
        self.factorize()              # get L
        self.forward_substitute()     # get y
        self.backward_substitute()    # get x
        
        end_time = time.time()       
        self.execution_time = end_time - start_time   # store execution time
        self.num_iterations = len(self.steps)        # store number of recorded steps
        
        return self.x


