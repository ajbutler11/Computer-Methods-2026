# region imports
from NumericalMethods import Secant
from math import cos  # needed for cos(2x)


# endregion

# region function definitions
def fn1(x):
    """x - 3cos(x) = 0"""
    return x - 3.0 * cos(x)


def fn2(x):
    """cos(2x) * x^3 = 0"""
    return cos(2.0 * x) * (x ** 3)


def main():
    """
       fn1:  x-3cos(x)=0; with x0=1, x1= 2, maxiter = 5 and xtol = 1e-4
       fn2:  cos(2x)*x**3=0; with x0=1, x1= 2, maxiter = 15 and xtol = 1e-8
       fn2:  cos(2x)*x**3=0; with x0=1, x1= 2, maxiter = 3 and xtol = 1e-8

       I observe that for functions 2 and 3, the answer should be pi/2 or about 1.57
    :return: nothing, just print results
    """
    r1 = Secant(fn1, 1, 2, 5, 1e-4)
    r2 = Secant(fn2, 1, 2, 15, 1e-8)
    r3 = Secant(fn2, 1, 2, 3, 1e-8)

    print("root of fn1 = {root:0.4f}, after {iter:0d} iterations".format(root=r1[0], iter=r1[1]))
    print("root of fn2 = {root:0.4f}, after {iter:0d} iterations".format(root=r2[0], iter=r2[1]))
    print("root of fn2 = {root:0.4f}, after {iter:0d} iterations".format(root=r3[0], iter=r3[1]))


# endregion

if __name__ == "__main__":
    main()