import time
from pivot_scaling import apply_partial_pivoting, apply_scaling

def gauss_elimination(A, b):
    """
    Solves Ax = b using Gaussian Elimination.
    Pivoting and scaling are applied using external helper functions
    implemented by noor.
    """

    start = time.time()

    # Convert values to float for safe numerical operations
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b)]
    n = len(A)

    # Apply scaling when noor enables this feature
    apply_scaling(A, b)

    # Forward Elimination
    for k in range(n):

        # Apply partial pivoting through noor's function
        apply_partial_pivoting(A, b, k)

        # Zero out rows below the pivot
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            A[i][k] = 0
            for j in range(k + 1, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # Back Substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]

    end = time.time()
    return x, (end - start) * 1000
