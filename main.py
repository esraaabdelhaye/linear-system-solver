from methods.iterative_method import iterative_method

def main():
    # Example system A * x = B
    A = [
        [12,3,-5],
        [1,5,3],

        [3,7,13]
    ]

    B = [1,28,76]

    X0 = [1,0,1]  # initial guess
    iterations = 6
    tol = 1e-1

    # Test Jacobi
    print("Testing Jacobi:")
    solver_jacobi = iterative_method(X0, A, B,6, iterations, tol, jacobi=True)
    result_jacobi = solver_jacobi.solve()
    print("Result:", result_jacobi)

    # Test Gauss-Seidel
    print("\nTesting Gauss-Seidel:")
    solver_gs = iterative_method(X0, A, B,6, iterations, tol, jacobi=False)
    result_gs = solver_gs.solve()
    print("Result:", result_gs)

if __name__ == "__main__":
    main()