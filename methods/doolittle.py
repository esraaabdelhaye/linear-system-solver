import numpy as np
import time

class LUDecomp:
    def __init__(self, a, b, n, tol, er, x = []):
        self.a = a
        self.b = b
        self.n = n
        self.x = x
        self.tol = tol
        self.er = 0
        self.o = [0] * n
        self.x = [0] * n

    def solve(self):
        self.decompose()
        if(self.er != -1):
            self.substitute(self.a, self.o, self.n, self.b, self.x)


    def decompose(self): 
        # for i in range(self.n): #scaling factors
        #     self.o[i] = i
        #     self.s[i] = abs(self.a[i, 1])
        #     for j in range(1, self.n):
        #         if(abs(self.a[i, j] > self.s[i])):
        #             self.s[i] = abs[i, j]


        for k in range(self.n-1):
            pivot(self.a, self.o, self.s, self.n, k)
            if abs(self.a[self.o[k],k]) / self.s[self.o[k]] < self.tol:
                er = -1
                return
            
            for i in range(k, self.n):
                factor = self.a[self.o[i],k] / self.a[self.o[k],k]
                self.a[self.o[i],k] = factor
                for j in range (k, self.n):
                    self.a[self.o[i],j] = self.a[self.o[i],j] - factor * self.a[self.o[k],j]
        if abs(self.a[self.o[self.n], self.n]) / self.s[self.o[self.n]] < self.tol:
            er = -1

    def substitute(a, o, n, b, x):
        y = [0] * n
        y[o[1]] = b[o[1]]
        for i in range (1, n): 
            sum = b[o[i]]
            for j in range (i-1):
                sum = sum - a[o[i],j] * b[o[j]]
            y[o[i]] = sum

        x[n] = y[o[n]] / a[o[n],n]
        for i in range (n-2, -1, -1):
            sum = 0
            for j in range(i+1, n):
                sum = sum + a[o[i],j] * x[j]
                x[i] = (y[o[i]] - sum) / a[o[i],i]
        
