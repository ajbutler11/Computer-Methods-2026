# problem2.py


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def circle_parabola(vals, x1, y1, R, width, offset):
    """
    System for intersection of:
    circle: (y-y1)^2 + (x-x1)^2 = R^2
    parabola: y = width * x^2 + offset
    :param vals: (x, y)
    :return: (f1, f2) = 0 at intersection
    """

    x, y = vals
    f1 = (y - y1)**2 + (x - x1)**2 -R**2
    f2 = width * x**2 + offset - y
    return (f1, f2)

# end region

def main():
    """
    Find and plot the meeting point(s) of a circle and a parabola using fsolve.
    Default test: center at (x1, y1) = (1,0), radius R = 4,
    parabola y = 0.5 x^2 +1
    """

    # default parameters
    x1_default = 1.0
    y1_default = 0.0
    R_default = 4.0
    width_default = 0.5
    offset_default = 1.0

    # optional user input (press Enter to keep defaults)

    try:
        s = input(f"Circle center x1? ({x1_default}): ").strip()
        x1 = x1_default if s == "" else float(s)

        s = input(f"Circle center y1? ({y1_default}): ").strip()
        y1 = y1_default if s == "" else float(s)

        s = input(f"Circle radius R? ({R_default}): ").strip()
        R = R_default if s == "" else float(s)

        s = input(f"Parabola width? ({width_default}): ").strip()
        width = width_default if s == "" else float(s)

        s = input(f"Parabola offset? ({offset_default}): ").strip()
        offset = offset_default if s == "" else float(s)
    except ValueError:
        print("bad input; using all default values.")
        x1, y1, r = x1_default, y1_default, R_default
        width, offset = width_default, offset_default

    # Solve for intersections
    guess_list = [(x1 + R/2, y1 + R/2), (x1 - R/2, y1 + R/2)]
    solutions = []

    for guess in guess_list:
        sol = fsolve(circle_parabola, guess, args=(x1, y1, R, width, offset))
        x_sol, y_sol = sol

        # no duplicates
        if not any(np.allclose(sol, s) for s in solutions):
            solutions.append(sol)

    print("\nIntersection Points:")
    for i, (xs, ys) in enumerate(solutions):
        print(f" Point {i+1}: x = {xs:0.4f}, y = {ys:0.4f}")

    # plotting from -10 to 10
    x_vals = np.linspace(-10, 10, 400)

    # solve for y from (y-y1)^2 = R^2 - (x - x1)^2
    rhs = R**2 - (x_vals - x1)**2
    rhs[rhs < 0] = np.nan # outside circle projection
    y_circle_top = y1 + np.sqrt(rhs)
    y_circle_bottom = y1 - np.sqrt(rhs)

    # parabola
    y_parab = width * x_vals**2 + offset

    plt.figure()
    plt.plot(x_vals, y_circle_top, 'b', label = 'circle')
    plt.plot(x_vals, y_circle_bottom, 'b')
    plt.plot(x_vals, y_parab, 'g', label = 'parabola')

    # plot intersection points
    for xs, ys in solutions:
        plt.plot(xs, ys, marker='o', markerfacecolor = 'none',
                 markeredgecolor = 'blue', markersize = 5,)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Intersection of circle and parabola")
    plt.show()

# end region

if __name__ == "__main__":
    main()