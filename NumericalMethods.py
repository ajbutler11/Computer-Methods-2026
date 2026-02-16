#region imports
import Gauss_Elim as GE  # this is the module that has useful matrix manipulation functions
from math import sqrt, pi, exp
#endregion


#region function definitions
def Probability(PDF, args, c, GT=True):
    """
    Return P(X > c) if GT is True, otherwise P(X < c), using Simpson's rule over a finite normal range.
    """
    mu, sig = args

    # choose integration limits: mu ± 5σ catches essentially all probability mass
    left = mu - 5 * sig
    right = mu + 5 * sig

    if GT:
        # probability to the right of c
        a = c
        b = right
    else:
        # probability to the left of c
        a = left
        b = c

    # clamp so we never integrate “backwards”
    if a < left:
        a = left
    if b > right:
        b = right
    if a >= b:
        return 0.0

    p = Simpson(PDF, (mu, sig, a, b))
    return p


def GPDF(args):
    """
    Evaluate the Gaussian PDF at x for mean mu and standard deviation sig.
    """
    x, mu, sig = args
    fx = (1.0 / (sig * sqrt(2.0 * pi))) * exp(-0.5 * ((x - mu) / sig) ** 2)
    return fx


def Simpson(fn, args, N=100):
    """
    Apply Simpson's 1/3 rule to integrate fn from a to b with args = (mu, sig, a, b).
    """
    mu, sig, a, b = args

    # ensure N is even
    if N % 2 == 1:
        N += 1

    h = (b - a) / float(N)

    fsum_odd = 0.0
    fsum_even = 0.0

    # interior points: i = 1..N-1
    for i in range(1, N):
        x = a + i * h
        fx = fn((x, mu, sig))
        if i % 2 == 1:
            fsum_odd += fx
        else:
            fsum_even += fx

    fa = fn((a, mu, sig))
    fb = fn((b, mu, sig))

    area = (h / 3.0) * (fa + fb + 4.0 * fsum_odd + 2.0 * fsum_even)
    return area


def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Use the Secant Method to approximate a root of fcn(x) near x0 and x1.
    """
    f0 = fcn(x0)
    f1 = fcn(x1)
    iter_count = 0  # track iterations

    for k in range(maxiter):
        iter_count += 1
        denom = f1 - f0
        if denom == 0.0:
            return x1, iter_count  # return (root, iterations)

        x2 = x1 - f1 * (x1 - x0) / denom
        if abs(x2 - x1) < xtol:
            return x2, iter_count  # return (root, iterations)

        x0, x1 = x1, x2
        f0, f1 = f1, fcn(x2)

    return x1, iter_count  # return (root, iterations) after maxiter


def GaussSeidel(Aaug, x, Niter=15):
    """
    Use Gauss-Seidel method to solve Ax=b from augmented matrix [A|b].
    """
    # Step 1: make diagonally dominant (call in-place, don't assign)
    GE.MakeDiagDom(Aaug)

    n_rows = len(Aaug)
    n_cols = len(Aaug[0])  # last column is b

    for iteration in range(Niter):
        for i in range(n_rows):
            # right hand side = b value for this equation
            rhs = Aaug[i][n_cols - 1]

            # subtract all other terms A[i,j]*x[j] where j != i
            for j in range(n_cols - 1):  # all A columns
                if j != i:
                    rhs -= Aaug[i][j] * x[j]

            # solve for x[i]: x[i] = rhs / A[i,i]
            x[i] = rhs / Aaug[i][i]

    return x


def main():
    '''
    This is a function I created for testing the numerical methods locally.
    :return: None
    '''
    #region testing GPDF
    fx = GPDF((0, 0, 1))
    print("{:0.5f}".format(fx))  # for N(0,1), f(0) ~ 0.39894
    #endregion

    #region testing Simpson
    p = Simpson(GPDF, (0, 1, -5, 0))  # should be close to 0.5
    print("Simpson p={:0.5f}".format(p))
    #endregion

    #region testing Probability
    p1 = Probability(GPDF, (0, 1), 0, True)  # P(X > 0) for N(0,1) ~ 0.5
    print("p1={:0.5f}".format(p1))
    #endregion

    # simple Secant test on f(x) = x^2 - 2
    def ftest(x):
        return x * x - 2.0

    root, it = Secant(ftest, 1.0, 2.0)
    print("Secant root ~ sqrt(2):", root, "in", it, "iters")

    # simple Gauss–Seidel test (2x2 system)
    Aaug = [[4.0, 1.0, 9.0],
            [2.0, 3.0, 13.0]]
    x0 = [0.0, 0.0]
    x_gs = GaussSeidel(Aaug, x0, Niter=10)
    print("Gauss–Seidel x:", x_gs)

#endregion


#region function calls
if __name__ == '__main__':
    main()
#endregion