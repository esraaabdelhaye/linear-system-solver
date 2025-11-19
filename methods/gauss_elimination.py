import time

def gauss_elimination(A, b):
    start = time.time()

    n = len(A)
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b))

    for k in range(n):
        max_row = max(range(k, n), key=lambda r: abs(A[r][k]))
        if A[max_row][k] == 0:
            return None, (time.time() - start) * 1000

        if max_row != k:
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            A[i][k] = 0
            for j in range(k + 1, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        if A[i][i] == 0:
            return None, (time.time() - start) * 1000
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]

    return x, (time.time() - start) * 1000
