import timeit
from AbstractSolver import AbstractSolver

class Doolittle(AbstractSolver):
    def __init__(self, A, b, precision=6, single_step=False):
        super().__init__(A, b, precision, single_step)
        self.x = [0] * self.n
        self.tol = 1e-9
        self.er = 0
        self.o = list(range(self.n))
        self.s = [0] * self.n
        self.a = self.A.copy()

    def solve(self):
        startTime = timeit.default_timer()
        self.validate()
        self.decompose()
        if(self.er == -1):
            return
        self.substitute()

        endTime = timeit.default_timer()
        time = endTime - startTime
        return self.x, time


    def decompose(self): 
        #getting scaling matrix
        for i in range(self.n): 
            self.o[i] = i
            self.s[i] = abs(self.a[i, 0])
            for j in range(1, self.n):
                if abs(self.a[i, j]) > self.s[i]:
                    self.s[i] = abs(self.a[i, j])


        for k in range(self.n-1):
            self.pivot(self.a, self.o, self.s, self.n, k)
            #check for singularity or near singularity
            if super().round_sig_fig(abs(self.a[self.o[k],k]) / self.s[self.o[k]]) < self.tol:
                self.er = -1
                return
            
            for i in range(k+1, self.n):
                factor = super().round_sig_fig(self.a[self.o[i],k] / self.a[self.o[k],k])
                #reuse A to store L and U instead of creating 2 spearate matrices
                self.a[self.o[i],k] = factor

                #eliminate coeffecients at column j in subsequent rows
                for j in range (k+1, self.n):
                    self.a[self.o[i],j] = super().round_sig_fig(self.a[self.o[i],j] - factor * self.a[self.o[k],j])
        #check for singularity (n-1) was not checked
        if super().round_sig_fig(abs(self.a[self.o[self.n - 1], self.n - 1])) < self.tol:
            self.er = -1

    def substitute(self):
        a, o, n, b, x = self.a, self.o, self.n, self.b, self.x
        y = [0] * n
        y[o[0]] = b[o[0]]
        for i in range (1, n): 
            sum = b[o[i]]
            for j in range (i):
                sum = super().round_sig_fig(sum - a[o[i],j] * y[o[j]])
            y[o[i]] = sum

        x[n-1] = y[o[n-1]] / a[o[n-1],n-1]
        for i in range (n-2, -1, -1):
            sum = 0
            for j in range(i+1, n):
                sum = super().round_sig_fig(sum + a[o[i],j] * x[j])
            x[i] = super().round_sig_fig((y[o[i]] - sum) / a[o[i],i])

    #find the largest coeffecient in a column after scaling
    def pivot(self, a, o, s, n, k):
        p = k 
        big = super().round_sig_fig(abs(a[o[k],k]) / s[o[k]])
        for i in range(k+1, n):
            dummy = super().round_sig_fig(abs(a[o[i],k] / s[o[i]]))
            if (dummy > big):
                big = dummy
                p = i
        #swap the order of the pivot row and the original row in the order
        #matrix instead of swapping rows of the actual matrix
        dummy = o[p]
        o[p] = o[k]
        o[k] = dummy

        
