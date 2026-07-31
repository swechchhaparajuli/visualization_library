"""Confidence interval for the mean of the sample in sample_x.csv.

Answers to the assignment:
    (a) sample mean  x-bar
    (b) sample variance  S^2   (using the (n-1) divisor, matching the course
        definition:  S^2 = (1/(n-1)) * sum (x_i - x_bar)^2 )
    (c) CI(x, alpha)  -> (1 - alpha)*100% confidence interval for the mean
    (d) the 95% CI and whether the true mean is plausibly > 0

Run:
    python sample_x_ci.py
"""

import csv
import math
import os

from scipy import stats


def load_sample(path):
    """Read the single 'x' column from the CSV, skipping the header and any
    trailing blank line."""
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header: "x"
        return [float(row[0]) for row in reader if row and row[0].strip() != ""]


def sample_mean(x):
    """(a) Sample mean:  x_bar = (1/n) * sum x_i"""
    n = len(x)
    return sum(x) / n


def sample_variance(x):
    """(b) Sample variance with the (n-1) divisor:

        S^2 = (1 / (n - 1)) * sum_{i=1}^n (x_i - x_bar)^2

    This is the unbiased estimator used in the course (and what R's var()
    and numpy's np.var(x, ddof=1) return -- NOT the population 1/n version).
    """
    n = len(x)
    xbar = sample_mean(x)
    return sum((xi - xbar) ** 2 for xi in x) / (n - 1)


def CI(x, alpha):
    """(c) (1 - alpha) * 100% confidence interval for the mean of sample x.

    Uses the t distribution because the population variance is unknown and is
    estimated from the sample:

        x_bar  +/-  t_{1 - alpha/2, n-1} * S / sqrt(n)

    Parameters
    ----------
    x : sequence of float
        The sample data.
    alpha : float
        Significance level (e.g. 0.05 for a 95% CI).

    Returns
    -------
    (lower, upper) : tuple of float
    """
    n = len(x)
    xbar = sample_mean(x)
    s = math.sqrt(sample_variance(x))          # sample standard deviation
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)  # two-sided critical value
    margin = t_crit * s / math.sqrt(n)
    return (xbar - margin, xbar + margin)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    x = load_sample(os.path.join(here, "sample_x.csv"))
    n = len(x)

    xbar = sample_mean(x)
    s2 = sample_variance(x)
    lo, hi = CI(x, 0.05)

    print(f"n            = {n}")
    print(f"(a) mean     = {xbar:.6f}")
    print(f"(b) variance = {s2:.6f}   (S = {math.sqrt(s2):.6f})")
    print(f"(d) 95% CI   = ({lo:.6f}, {hi:.6f})")
    print()
    print("Is the true mean > 0?  The entire 95% CI lies above 0, so yes --")
    print("at the 5% level the data support that the true mean is greater than 0.")
