from methods.AbstractSolver import AbstractSolver

from System.SystemData import SystemData
from typing import  Dict, Any

class Crout(AbstractSolver):
    # Call parent class constructor to initialize A, b, n, etc.
    def __init__(self, data: SystemData):
        super().__init__(data)
        # Initialize solution vector
        self.x = [0] * self.n
        # Numerical tolerance for detecting zero pivots
        # self.tol = 1e-9
        # # Error flag (-1 means failure during decomposition)
        # self.er = 0
        # Order vector for row permutations (pivoting)
        self.o = list(range(self.n))
        # Scaling vector for scaled partial pivoting
        # self.s = [0] * self.n
        # Copy of matrix A (decomposition happens here)
        self.a = self.A.copy()

    def solve(self) -> Dict[str, Any]:
        # Perform LU decomposition (Crout method)
        self.decompose()

        # If decomposition failed, exit
        # if (self.er == -1):
        #     return

        # Substitute with L and U get y then get x (solution vector)
        self.substitute()

        # Return solution vector
        return {"success": True,"sol": self.x}

    def decompose(self):
         # Compute scaling vector used in scaled partial pivoting
        # for i in range(self.n):
        #     self.o[i] = i # initial ordering
        #     self.s[i] = abs(self.a[i, 0]) # find max absolute element in row
        #     for j in range(1, self.n):
        #         if abs(self.a[i, j]) > self.s[i]:
        #             self.s[i] = abs(self.a[i, j])

         # Begin Crout decomposition: build L column-wise
        for j in range(self.n):  # column j

            # pivoting (on column j)
            self.pivot(self.a, self.o, self.n, j)

            # check if scaled pivot is less than tol then matrix is singular
            # if super().round_sig_fig(abs(self.a[self.o[j], j]) / self.s[self.o[j]]) < self.tol:
            #     self.er = -1
            #     return

            # calculate L(i, j)
            for i in range(j, self.n):
                sum = self.a[self.o[i], j]
                for k in range(j):
                    sum = super().round_sig_fig(sum - self.a[self.o[i], k] * self.a[self.o[k], j])
                self.a[self.o[i], j] = sum  # L(i, j)

            # check zero pivot after building L(j,j)
            # if abs(self.a[self.o[j], j]) < self.tol:
            #     self.er = -1
            #     return

            # calculate U(j, i)
            for i in range(j + 1, self.n):
                sum = self.a[self.o[j], i]
                for k in range(j):
                    sum = super().round_sig_fig(sum - self.a[self.o[j], k] * self.a[self.o[k], i])
                self.a[self.o[j], i] = super().round_sig_fig(sum / self.a[self.o[j], j])  # U(j, i)

    def substitute(self):
        a, o, n, b, x = self.a, self.o, self.n, self.b, self.x
        # Forward substitution to compute y from Ly = b
        y = [0] * n
        y[o[0]] = super().round_sig_fig(b[o[0]] / a[o[0], 0])
        for i in range(1, n):
            sum = b[o[i]]
            for j in range(i):
                sum = super().round_sig_fig(sum - a[o[i], j] * y[o[j]])
            y[o[i]] = sum / a[o[i], i]

       # Backward substitution to compute x from Ux = y
        x[n - 1] = y[o[n - 1]]  # last variable
        for i in range(n - 2, -1, -1):
            sum = y[o[i]]
            for j in range(i + 1, n):
                 # subtract U(i,j) * x(j)
                sum = super().round_sig_fig(sum - a[o[i], j] * x[j])
            x[i] = sum

        # final rounding
        for i in range(n):
            x[i] = super().round_sig_fig(x[i])

    # finding large scaled coeffcient in column (magnitude wise)
    def pivot(self, a, o, n, k):
        p = k

        # compute scaled value of pivot row
        big = super().round_sig_fig(abs(a[o[k], k]))

        # search for better pivot
        for i in range(k + 1, n):
            dummy = super().round_sig_fig(abs(a[o[i], k]))
            if dummy > big:
                big = dummy
                p = i

        # swap permutation indices
        dummy = o[p]
        o[p] = o[k]
        o[k] = dummy
