# hw3b.py

from math import sqrt, pi, gamma

# t-distribution PDF (symmetric around 0)
def t_pdf(u, m):
    return (1.0 + (u*u)/m) ** (-(m+1)/2.0)

# Simpson's rule on [a, b] with n even subintervals
def simpson(f, a, b, n, *args):
    if n % 2 == 1:
        n += 1  # ensure even
    h = (b - a) / n
    s = f(a, *args) + f(b, *args)
    for k in range(1, n):
        xk = a + k * h
        s += (4 if k % 2 == 1 else 2) * f(xk, *args)
    return s * h / 3.0

# Normalization constant K_m
def K_m(m):
    return gamma((m + 1.0) / 2.0) / (sqrt(m * pi) * gamma(m / 2.0))

# CDF F(z) for t-distribution with m d.o.f.
def t_cdf(z, m, n_int=1000):
    K = K_m(m)
    if z == 0.0:
        return 0.5
    if z > 0:
        integral = simpson(t_pdf, 0.0, z, n_int, m)
        return 0.5 + K * integral
    else:  # z < 0, use symmetry
        integral = simpson(t_pdf, 0.0, -z, n_int, m)
        return 0.5 - K * integral

def main():
    print("t-distribution CDF calculator (hw3b)")
    try:
        m = int(input("Degrees of freedom m "))
        z = float(input("z value: "))
    except ValueError:
        print("Invalid input.")
        return

    if m <= 0:
        print("m must be a positive integer.")
        return

    prob = t_cdf(z, m)
    print(f"F({z:.4f}) for m = {m} ≈ {prob:.6f}")

if __name__ == "__main__":
    main()
