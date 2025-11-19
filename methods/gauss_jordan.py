import time
from pivot_scaling import apply_partial_pivoting, apply_scaling

def gauss_jordan(A, b):
    """
    Solves Ax = b using Gauss–Jordan Elimination.
    Pivoting and scaling are handled by helper functions implemented
    by noor.
    """

    start = time.time()

    # Convert to float
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b)]
    n = len(A)

    # Apply scaling (noor's function)
    apply_scaling(A, b)

    # Main Gauss–Jordan elimination
    for k in range(n):

        # Apply noor's pivoting function
        apply_partial_pivoting(A, b, k)

        # Normalize pivot row
        pivot = A[k][k]
        for j in range(k, n):
            A[k][j] /= pivot
        b[k] /= pivot

        # Eliminate all other rows
        for i in range(n):
            if i != k:
                factor = A[i][k]
                A[i][k] = 0
                for j in range(k + 1, n):
                    A[i][j] -= factor * A[k][j]
                b[i] -= factor * b[k]

    end = time.time()
    return b, (end - start) * 1000
