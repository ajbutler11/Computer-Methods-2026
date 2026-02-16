# region imports
from copy import deepcopy
from NumericalMethods import GaussSeidel
from Gauss_Elim import AugmentMatrix
# endregion

def main():
    """
    Use GaussSeidel to solve two systems of linear equations.
    """
    # System 1: 3x3
    # [ 3  1 -1 |  2]
    # [ 1  4  1 | 12]
    # [ 2  1  2 | 10]
    A1 = [[3.0, 1.0, -1.0],
          [1.0, 4.0, 1.0],
          [2.0, 1.0, 2.0]]
    b1 = [2.0, 12.0, 10.0]
    x1_init = [0.0, 0.0, 0.0]
    Aaug1 = AugmentMatrix(A1, b1)
    x1_sol = GaussSeidel(Aaug1, x1_init, Niter=15)
    print("System 1 (3x3): x1={:7.4f}, x2={:7.4f}, x3={:7.4f}".format(x1_sol[0], x1_sol[1], x1_sol[2]))

    # System 2: 4x4
    # [ 1 -10  2  4 |  2]
    # [ 3  1  4 12 | 12]
    # [ 9  2  3  4 | 21]
    # [-1  2  7  3 | 37]
    A2 = [[1.0, -10.0, 2.0, 4.0],
          [3.0, 1.0, 4.0, 12.0],
          [9.0, 2.0, 3.0, 4.0],
          [-1.0, 2.0, 7.0, 3.0]]
    b2 = [2.0, 12.0, 21.0, 37.0]
    x2_init = [0.0, 0.0, 0.0, 0.0]
    Aaug2 = AugmentMatrix(A2, b2)
    x2_sol = GaussSeidel(Aaug2, x2_init, Niter=15)
    print("System 2 (4x4): x1={:7.4f}, x2={:7.4f}, x3={:7.4f}, x4={:7.4f}".format(x2_sol[0], x2_sol[1], x2_sol[2], x2_sol[3]))


if __name__=="__main__":
    main()
