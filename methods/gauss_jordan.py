import time

def gauss_jordan(A, b):
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

        pivot = A[k][k]
        for j in range(k, n):
            A[k][j] /= pivot
        b[k] /= pivot

        for i in range(n):
            if i != k:
                factor = A[i][k]
                A[i][k] = 0
                for j in range(k + 1, n):
                    A[i][j] -= factor * A[k][j]
                b[i] -= factor * b[k]

    return b, (time.time() - start) * 1000
