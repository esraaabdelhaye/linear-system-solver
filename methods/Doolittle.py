import timeit
from methods.AbstractSolver import AbstractSolver
from System.SystemData import SystemData
from typing import  Dict, Any

class Doolittle(AbstractSolver):
    def __init__(self, data: SystemData):
        # Initialize the parent solver
        super().__init__(data)

        # Initialize solution vector
        self.x = [0] * self.n

        # Tolerance for detecting zero/near-zero pivots
        self.tol = 1e-9
        self.er = 0
        self.o = list(range(self.n))
        # self.s = [0] * self.n

        # Copy of A (used to store both L and U)
        self.a = self.A.copy()


    def solve(self) -> Dict[str, Any]:
        print("Solving Doolittle...")

        # Doolittle LU decomposition
        self.decompose()
        # if (self.er == -1):
        #     return
        
        # Forward + backward substitution
        self.substitute()
        return {"success": True,"sol": self.x}

    def decompose(self):
        # Build the scaling vector based on the largest absolute value in each row
        # for i in range(self.n):
        #     self.o[i] = i
        #     self.s[i] = abs(self.a[i, 0])
        #     for j in range(1, self.n):
        #         if abs(self.a[i, j]) > self.s[i]:
        #             self.s[i] = abs(self.a[i, j])

        # Main Doolittle loop for all columns except last
        for k in range(self.n-1):
             # Perform partial pivoting
            self.pivot(self.a, self.o, self.n, k)
            # check for singularity or near singularity
            # if super().round_sig_fig(abs(self.a[self.o[k], k]) / self.s[self.o[k]]) < self.tol:
            #     self.er = -1
            #     return
            
             # Eliminate entries below pivot
            for i in range(k + 1, self.n):
                factor = super().round_sig_fig(self.a[self.o[i], k] / self.a[self.o[k], k])
                # Store L(i,k) in A (lower-triangular)
                self.a[self.o[i], k] = factor

                # Update remaining elements in the row (U part)
                for j in range(k + 1, self.n):
                    self.a[self.o[i], j] = super().round_sig_fig(self.a[self.o[i], j] - factor * self.a[self.o[k], j])
        # Check final pivot (last diagonal element)
        if super().round_sig_fig(abs(self.a[self.o[self.n - 1], self.n - 1])) < self.tol:
            self.er = -1

    def substitute(self):
        a, o, n, b, x = self.a, self.o, self.n, self.b, self.x
        y = [0] * n

        # Forward substitution to solve Ly = b
        y[o[0]] = b[o[0]]
        for i in range(1, n):
            sum = b[o[i]]
            for j in range(i):
                # subtract L(i,j) * y(j)
                sum = super().round_sig_fig(sum - a[o[i], j] * y[o[j]])
            y[o[i]] = sum

        # Backward substitution to solve Ux = y
        x[n - 1] = y[o[n - 1]] / a[o[n - 1], n - 1]
        for i in range(n - 2, -1, -1):
            sum = 0
            for j in range(i + 1, n):
                # accumulate U(i,j) * x(j)
                sum = super().round_sig_fig(sum + a[o[i], j] * x[j])

            # Compute x(i)
            x[i] = super().round_sig_fig((y[o[i]] - sum) / a[o[i], i])

    # find the largest coeffecient in a column after scaling
    def pivot(self, a, o, n, k):
        p = k
        big = super().round_sig_fig(abs(a[o[k], k]))
        for i in range(k + 1, n):
            dummy = super().round_sig_fig(abs(a[o[i], k]))
            if (dummy > big):
                big = dummy
                p = i
        # swap the order of the pivot row and the original row in the order
        # matrix instead of swapping rows of the actual matrix
        dummy = o[p]
        o[p] = o[k]
        o[k] = dummy

