
import numpy as np
from Cholesky import Cholesky


def test_cholesky_case(name, A, b):
    print("\n========================================")
    print(f"TEST CASE: {name}")
    print("========================================")

    print("Matrix A:\n", A)
    print("Vector b:", b)

    # Expected solution using NumPy
    try:
        expected_x = np.linalg.solve(A.copy(), b.copy())
        print("\nExpected solution (NumPy):", expected_x)
    except Exception as e:
        expected_x = None
        print("\nNumPy solve FAILED:", e)

    # Run Cholesky
    print("\n--- Testing Cholesky Solver ---")
    try:
        solver = Cholesky(A.copy(), b.copy(), precision=6, single_step=False)
        result = solver.solve()
        x_ch = result["solution"]
        print("Cholesky solution:", x_ch)
    except Exception as e:
        x_ch = None
        print("Cholesky error:", e)

    # Compare accuracy
    if expected_x is not None and x_ch is not None:
        diff = np.linalg.norm(expected_x - x_ch)
        print("Difference from expected:", diff)

    print("========================================\n")


def main():
    print("\n========================================")
    print("Starting Cholesky Test Cases...")
    print("========================================")

    # 1. Simple symmetric positive-definite matrix
    A1 = np.array([[4, 2],
                   [2, 3]], float)
    b1 = np.array([6, 5], float)
    test_cholesky_case("2x2 SPD matrix", A1, b1)

    # 2. 3x3 SPD matrix
    A2 = np.array([[25, 15, -5],
                   [15, 18,  0],
                   [-5, 0, 11]], float)
    b2 = np.array([35, 33, 6], float)
    test_cholesky_case("3x3 SPD matrix", A2, b2)

    # 3. Random SPD matrix (A = MᵀM)
    np.random.seed(1)
    M = np.random.rand(4, 4)
    A3 = M.T @ M + np.eye(4)*1e-3     # Ensure SPD
    b3 = np.random.rand(4)
    test_cholesky_case("Random 4x4 SPD", A3, b3)

    # 4. NON-SPD matrix (should fail)
    A4 = np.array([[1, 2],
                   [2, 1]], float)    # Not positive-definite
    b4 = np.array([3, 3], float)
    test_cholesky_case("Non-SPD matrix (expect failure)", A4, b4)

    # 5. Larger SPD 5x5
    M = np.random.rand(5, 5)
    A5 = M.T @ M + np.eye(5)*1e-3
    b5 = np.random.rand(5)
    test_cholesky_case("Random 5x5 SPD", A5, b5)

    print("\n========================================")
    print("All Cholesky tests completed.")
    print("========================================")


if __name__ == "__main__":
    main()
