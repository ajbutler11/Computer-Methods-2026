#region imports
from math import sqrt, pi, exp
from NumericalMethods import GPDF, Simpson, Probability
#endregion


#region root finding: secant method
def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method to find a root of f starting from x0, x1.
    """
    f0 = f(x0)
    f1 = f(x1)
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-14:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)
    return x1
#endregion


#region helper probability functions
def central_probability(mu, sigma, c):
    """
    P(mu - (c-mu) < x < mu + (c-mu) | N(mu, sigma))
    """
    half_width = c - mu
    a = mu - half_width
    b = mu + half_width
    # P(a < x < b) = P(x < b) - P(x < a)
    p_b = Probability(GPDF, (mu, sigma), b, GT=False)
    p_a = Probability(GPDF, (mu, sigma), a, GT=False)
    return p_b - p_a


def outside_probability(mu, sigma, c):
    """
    P(x < mu - (c-mu) or x > mu + (c-mu) | N(mu, sigma))
    = 1 - central_probability
    """
    p_center = central_probability(mu, sigma, c)
    return 1.0 - p_center
#endregion


#region homework 2a cases (unchanged)
def run_homework_cases():
    """
    Compute and print the two normal probabilities required in hw2a.
    """
    # First required value: P(x<105|N(100,12.5))
    mu1 = 100.0
    stDev1 = 12.5
    c1 = 105.0
    p1 = Probability(GPDF, (mu1, stDev1), c1, GT=False)

    # Second required value: P(x>μ+2σ|N(100, 3)) where μ+2σ = 100 + 2*3 = 106
    mu2 = 100.0
    stDev2 = 3.0
    c2 = mu2 + 2.0 * stDev2
    p2 = Probability(GPDF, (mu2, stDev2), c2, GT=True)

    print("P(x<{:.2f}|N({:.0f},{:.1f}))={:.2f}".format(c1, mu1, stDev1, p1))
    print("P(x>{:.2f}|N({:.0f},{:.0f}))={:.2f}".format(c2, mu2, stDev2, p2))
#endregion


#region main interactive logic
def main():
    # run hw2a cases first
    run_homework_cases()

    print("\nNow enter your own normal probability problem.")
    try:
        mean = float(input("Population mean μ? "))
        stDev = float(input("Standard deviation σ? "))
    except ValueError:
        print("Could not convert mean or standard deviation to numbers.")
        return

    mode = input("Are you specifying c and seeking P, or specifying P and seeking c? (enter 'c->P' or 'P->c'): ").strip().lower()

    # Ask which case (single or double sided, inside or outside)
    print("\nChoose probability type:")
    print("  1) P(x < c | μ, σ)")
    print("  2) P(x > c | μ, σ)")
    print("  3) P(μ-(c-μ) < x < μ+(c-μ) | μ, σ)   [central, double-sided]")
    print("  4) P(x < μ-(c-μ) OR x > μ+(c-μ) | μ, σ) [outside, two tails]")
    case = input("Enter 1, 2, 3, or 4: ").strip()

    if mode == "c->p":
        # user gives c, we compute P
        try:
            c = float(input("Enter c: "))
        except ValueError:
            print("Could not convert c to a number.")
            return

        if case == "1":
            GT = False
            p = Probability(GPDF, (mean, stDev), c, GT)
            print("P(x<{:.4f}|N({:.4f},{:.4f})) = {:.6f}".format(c, mean, stDev, p))
        elif case == "2":
            GT = True
            p = Probability(GPDF, (mean, stDev), c, GT)
            print("P(x>{:.4f}|N({:.4f},{:.4f})) = {:.6f}".format(c, mean, stDev, p))
        elif case == "3":
            p = central_probability(mean, stDev, c)
            print("P({:.4f} < x < {:.4f}|N({:.4f},{:.4f})) = {:.6f}".format(
                mean - (c - mean), mean + (c - mean), mean, stDev, p))
        elif case == "4":
            p = outside_probability(mean, stDev, c)
            print("P(x < {:.4f} OR x > {:.4f}|N({:.4f},{:.4f})) = {:.6f}".format(
                mean - (c - mean), mean + (c - mean), mean, stDev, p))
        else:
            print("Unrecognized case selection.")

    elif mode == "p->c":
        # user gives P, we solve for c using Secant method
        try:
            P_target = float(input("Desired probability (between 0 and 1)? "))
        except ValueError:
            print("Could not convert probability to a number.")
            return

        if not (0.0 < P_target < 1.0):
            print("Probability must be between 0 and 1 (exclusive).")
            return

        # define F(c) for each case
        if case == "1":
            # left tail
            def F(c):
                return Probability(GPDF, (mean, stDev), c, GT=False) - P_target

        elif case == "2":
            # right tail
            def F(c):
                return Probability(GPDF, (mean, stDev), c, GT=True) - P_target

        elif case == "3":
            # central symmetric
            def F(c):
                return central_probability(mean, stDev, c) - P_target

        elif case == "4":
            # outside symmetric (two tails)
            def F(c):
                return outside_probability(mean, stDev, c) - P_target

        else:
            print("Unrecognized case selection.")
            return

        # initial guesses around mean: one sigma and two sigma away
        c0 = mean - stDev
        c1 = mean + stDev

        c_root = secant(F, c0, c1, tol=1e-6, max_iter=100)

        # Report result
        if case in ["1", "2"]:
            kind = "<" if case == "1" else ">"
            print("c ≈ {:.6f} such that P(x{}c|N({:.4f},{:.4f})) ≈ {:.6f}".format(
                c_root, kind, mean, stDev, P_target))
        elif case == "3":
            a = mean - (c_root - mean)
            b = mean + (c_root - mean)
            print("c ≈ {:.6f} giving central interval ({:.6f}, {:.6f}) with probability ≈ {:.6f}".format(
                c_root, a, b, P_target))
        elif case == "4":
            a = mean - (c_root - mean)
            b = mean + (c_root - mean)
            print("c ≈ {:.6f} giving outside region x<{:.6f} or x>{:.6f} with probability ≈ {:.6f}".format(
                c_root, a, b, P_target))

    else:
        print("Unrecognized mode. Please enter 'c->P' or 'P->c'.")
#endregion


#region function calls
if __name__ == "__main__":
    main()
#endregion