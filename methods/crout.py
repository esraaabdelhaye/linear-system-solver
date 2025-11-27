import numpy as np
import timeit
from doolittle import Doolittle

class Crout:
    def __init__(self, a, b, n, x, tol):
        self.doolittle = Doolittle(self, a, b, n, tol, er, x = [])
        self.a = a
        self.b = b
        self.n = n
        self.x = x
        self.tol = tol
        self.er = 0
        self.o = [0] * n
        self.x = [0] * n

    def solve(self):
        startTime = timeit.default_timer()
        self.decompose()
        a = a.T
        if(self.er != -1):
            self.substitute(self.a, self.o, self.n, self.b, self.x)
        else: return

        endTime = timeit.default_timer()
        time = endTime - startTime

        return self.x

    def decompose(self):
        self.doolittle.decompose()

    def substitute(a, o, n, b, x):
        #forward sub
        y = [0] * n
        y[o[0]] = b[o[0]] / a[o[0], 0]
        for i in range (1, n): 
            sum = b[o[i]]
            for j in range (i):
                sum = sum - a[o[i],j] * y[o[j]]
            y[o[i]] = sum / a[o[i],i]

        #backward sub
        x[n-1] = y[o[n-1]] 
        for i in range (n-2, -1, -1):
            sum = 0
            for j in range(i+1, n):
                sum = sum + a[o[i],j] * x[j]
            x[i] = (y[o[i]] - sum) 




#Lcrout = Utdoolittle
#Ucrout = Ltdoolittle

# def getTranspose(A):
#     rows = len(A)
#     cols = len(A[0])

#     #initializing values of A transpose and determining no. of rows and cols
#     At = [[0] * rows for _ in range(cols)]

#     for i in range(rows):
#         for j in range(cols):
#             At[j][i] = A[i][j]

#     return At