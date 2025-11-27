import numpy as np
import timeit

class Doolittle:
    def __init__(self, a, b, n, tol):
        self.a = a
        self.b = b
        self.n = n
        self.x = [0] * n
        self.tol = tol
        self.er = 0
        self.o = [0] * n
        self.x = [0] * n
        self.s = [0] * n

    def solve(self):
        startTime = timeit.default_timer()
        self.decompose()
        if(self.er == -1):
            return
        self.substitute()

        endTime = timeit.default_timer()
        time = endTime - startTime

        return self.x


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
            if abs(self.a[self.o[k],k]) / self.s[self.o[k]] < self.tol:
                self.er = -1
                return
            
            for i in range(k+1, self.n):
                factor = self.a[self.o[i],k] / self.a[self.o[k],k]
                self.a[self.o[i],k] = factor
                for j in range (k+1, self.n):
                    self.a[self.o[i],j] = self.a[self.o[i],j] - factor * self.a[self.o[k],j]
        if abs(self.a[self.o[self.n - 1], self.n - 1]) < self.tol:
            self.er = -1

    def substitute(self):
        a, o, n, b, x = self.a, self.o, self.n, self.b, self.x
        y = [0] * n
        y[o[0]] = b[o[0]]
        for i in range (1, n): 
            sum = b[o[i]]
            for j in range (i):
                sum = sum - a[o[i],j] * y[o[j]]
            y[o[i]] = sum

        x[n-1] = y[o[n-1]] / a[o[n-1],n-1]
        for i in range (n-2, -1, -1):
            sum = 0
            for j in range(i+1, n):
                sum = sum + a[o[i],j] * x[j]
            x[i] = (y[o[i]] - sum) / a[o[i],i]

    def pivot(self, a, o, s, n, k):
        p = k 
        big = abs(a[o[k],k]) / s[o[k]]
        for i in range(k+1, n):
            dummy = abs(a[o[i],k] / s[o[i]])
            if (dummy > big):
                big = dummy
                p = i
        dummy = o[p]
        o[p] = o[k]
        o[k] = dummy

        
