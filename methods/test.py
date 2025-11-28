import numpy as np
from doolittle import Doolittle
from crout import Crout

def test_case(name, A, b, tol=1e-9):
    n = len(A)
    print(f"\n==============================")
    print(f"TEST CASE: {name}")
    print(f"==============================")
    print("Matrix A:")
    print(A)
    print("Vector b:", b)

    # Expected solution using NumPy
    try:
        expected_x = np.linalg.solve(A.copy(), b.copy())
        print("\nExpected solution (NumPy):", expected_x)
    except Exception as e:
        print("\nNumPy failed:", e)
        expected_x = None

    # ---------- Test Doolittle ----------
    print("\n--- Testing DOOLITTLE ---")
    try:
        d = Doolittle(A.copy(), b.copy(), tol)
        x_d = d.solve()
        print("Doolittle solution:", x_d)
    except Exception as e:
        print("Doolittle error:", e)
        x_d = None

    if expected_x is not None and x_d is not None:
        diff = np.linalg.norm(expected_x - x_d)
        print("Difference from expected:", diff)

    # ---------- Test Crout ----------
    print("\n--- Testing CROUT ---")
    try:
        c = Crout(A.copy(), b.copy(), tol)
        x_c = c.solve()
        print("Crout solution:", x_c)
    except Exception as e:
        print("Crout error:", e)
        x_c = None

    if expected_x is not None and x_c is not None:
        diff = np.linalg.norm(expected_x - x_c)
        print("Difference from expected:", diff)


def main():
    print("\n========================================")
    print("Starting LU Decomposition Test Cases...")
    print("========================================")

    # Test Case 1: Simple 3x3 system
    A1 = np.array([[2, 1, 1],
                   [4, -6, 0],
                   [-2, 7, 2]], dtype=float)
    b1 = np.array([5, -2, 9], dtype=float)
    test_case("Simple 3x3", A1, b1)

    # Test Case 2: 4x4 well-conditioned matrix
    A2 = np.array([[1, 2, 3, 4],
                   [2, 5, 2, 1],
                   [1, 0, 3, 2],
                   [4, 1, 2, 3]], dtype=float)
    b2 = np.array([10, 8, 7, 9], dtype=float)
    test_case("4x4 well-conditioned", A2, b2)

    # Test Case 3: Random 5x5 system
    np.random.seed(0)
    A3 = np.random.rand(5, 5) * 10
    b3 = np.random.rand(5) * 10
    test_case("Random 5x5", A3, b3)

    # Test Case 4: Singular matrix
    A4 = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [1, 1, 1]], dtype=float)
    b4 = np.array([6, 12, 3], dtype=float)
    test_case("Singular Matrix", A4, b4)

    # Test Case 5: Nearly singular matrix (pivoting test)
    A5 = np.array([[1e-10, 1],
                   [1, 1]], dtype=float)
    b5 = np.array([1, 2], dtype=float)
    test_case("Nearly Singular Pivoting Stress Test", A5, b5)

    print("\n========================================")
    print("All tests completed.")
    print("========================================")


if __name__ == "__main__":
    main()