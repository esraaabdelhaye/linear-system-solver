import numpy as np

# solving Ax = B using gauss_seidel iterative method
# A => coefficients matrix
# B => constants matrix
# x0 => initial guess                      (default value: None)
# num_of_iterations => maximum number of iterations (default value: 10000)
# tol => tolerance                         (default value: 1e-6)

def gauss_seidel(A, b, x0=None, num_of_iterations=10000, tol=1e-6, stop_by_tolerance = True):
    A = np.array(A, dtype=float)
    B = np.array(b, dtype=float)
    n = len(b)

    if x0 is None:        # initial guess was not provided
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    for it in range(num_of_iterations):
        x_old = x.copy()

        for i in range(n):
            # these x values are the NEW updated ones because we’ve already computed them earlier in this iteration.
            sum1 = np.dot(A[i, :i], x[:i])
            # these x values are the OLD ones because we’ve computed in this iteration yet.
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (B[i] - sum1 - sum2) / A[i, i]

        # absolute relative error
        error = np.max(np.abs((x - x_old) / (x + 1e-12))) # Division by zero protection

        if stop_by_tolerance and error < tol :
            break

    return x


# if stop_by_tolerance = true then the function will stop when we reach an error smaller than our tolerance
# otherwise it will loop according to the number of iterations provided

# the expected behaviour is one of the following three cases
# (1) providing the num_of_iterations and the tol and setting stop_by_tolerance to true
#           ==> the function will stop when all the iterations are done or we reach the target tolerance
# (2) providing the num_of_iterations only and stop_by_tolerance is false
#           ==> the function will stop only when all the iterations are done
# (3) providing the tol only and setting stop_by_tolerance to true
#           ==> the function will stop when the target tolerance is reached (as long as the iterations number doesn't reach 10000)
