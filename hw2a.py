#region imports
from math import sqrt, pi, exp
from NumericalMethods import GPDF, Simpson, Probability
#endregion

#region function definitions
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

    # Print in the format shown in the assignment
    print("P(x<{:.2f}|N({:.0f},{:.1f}))={:.2f}".format(c1, mu1, stDev1, p1))
    print("P(x>{:.2f}|N({:.0f},{:.0f}))={:.2f}".format(c2, mu2, stDev2, p2))


def main():
    """
    First run the fixed homework cases, then let the user try their own normal probability.
    """
    # run the two specific cases required by the homework
    run_homework_cases()

    # --- original interactive test code (kept) ---
    #region testing user input
    # The following code solicites user input through the CLI.
    mean = input("Population mean? ")
    stDev = input("Standard deviation? ")
    c = input("c value? ")
    GT = True if input("Probability greater than c? ").lower() in ["y", "yes", "true"] else False
    print("P(x" + (">" if GT else "<") + c + "|N(" + mean + ", " + stDev + "))")

    # convert inputs to floats and call Probability for the user-defined case
    try:
        mean_f = float(mean)
        stDev_f = float(stDev)
        c_f = float(c)
        p_user = Probability(GPDF, (mean_f, stDev_f), c_f, GT)
        print("Result = {:.4f}".format(p_user))
    except ValueError:
        print("Could not convert your inputs to numbers.")
    #endregion
#endregion

#region function calls
if __name__ == "__main__":
    main()
#endregion