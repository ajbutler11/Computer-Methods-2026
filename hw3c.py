# hw3c.py

# region imports
from copy import deepcopy
from math import sqrt
from Gauss_Elim import AugmentMatrix
from DoolittleMethod import LUSolve
# endregion


def is_symmetric(A, tol=1e-12):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def is_positive_definite(A):
    """
    Check PD by attempting Cholesky factorization without actually returning L.
    """
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0.0:
                    return False
                L[i][i] = sqrt(val)
            else:
                if abs(L[j][j]) < 1e-15:
                    return False
                L[i][j] = (A[i][j] - s) / L[j][j]
    return True


def cholesky_factor(A):
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0.0:
                    raise ValueError("Matrix is not positive definite.")
                L[i][i] = sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def forward_substitution(L, b):
    n = len(L)
    y = [0.0]*n
    for i in range(n):
        s = sum(L[i][j]*y[j] for j in range(i))
        y[i] = (b[i] - s)/L[i][i]
    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0]*n
    for i in reversed(range(n)):
        s = sum(U[i][j]*x[j] for j in range(i+1, n))
        x[i] = (y[i] - s)/U[i][i]
    return x


def cholesky_solve(A, b):
    L = cholesky_factor(A)
    y = forward_substitution(L, b)
    n = len(L)
    U = [[L[j][i] for j in range(n)] for i in range(n)]  # U = L^T
    x = backward_substitution(U, y)
    return x


def solve_system(A, b):
    """
    Decide whether to use Cholesky or Doolittle, solve, and return (x, method_str).
    """
    if is_symmetric(A) and is_positive_definite(A):
        # Use Cholesky
        x = cholesky_solve(A, b)
        method = "Cholesky"
    else:
        x = LUSolve(A, b)
        method = "Doolittle"
    return x, method


def main():
    # Problem 1:
    #  x1 + x2 - 3x3 + 2x4 = 15
    # -x1 + 5x2 + 5x3       = -35
    #  3x1 - 5x2 +19x3 + 3x4 = 94
    #           2x3 + 2x4    = 21
    A1 = [
        [1.0,  1.0, -3.0,  2.0],
        [-1.0, 5.0,  5.0,  0.0],
        [3.0, -5.0, 19.0,  3.0],
        [0.0,  0.0,  2.0,  2.0],
    ]
    b1 = [15.0, -35.0, 94.0, 21.0]

    x1, method1 = solve_system(deepcopy(A1), b1)
    print("Problem 1 solution using {}:".format(method1))
    print("x1 = {:10.6f}, x2 = {:10.6f}, x3 = {:10.6f}, x4 = {:10.6f}".format(
        x1[0], x1[1], x1[2], x1[3]
    ))

    # Problem 2:
    #  4x1 - 2x2 + 4x3         = 20
    #  2x1 + 2x2 + 3x3 + 2x4   = 36
    #  4x1 + 3x2 + 6x3 + 3x4   = 60
    #           2x2 + 3x3 + 9x4 = 122
    A2 = [
        [4.0, -2.0, 4.0,  0.0],
        [2.0,  2.0, 3.0,  2.0],
        [4.0,  3.0, 6.0,  3.0],
        [0.0,  2.0, 3.0,  9.0],
    ]
    b2 = [20.0, 36.0, 60.0, 122.0]

    x2, method2 = solve_system(deepcopy(A2), b2)
    print("Problem 2 solution using {}:".format(method2))
    print("x1 = {:10.6f}, x2 = {:10.6f}, x3 = {:10.6f}, x4 = {:10.6f}".format(
        x2[0], x2[1], x2[2], x2[3]
    ))


if __name__ == "__main__":
    main()
