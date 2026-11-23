"""Starter for fitting/allocation (A, B, F1(a), F2), acquisition (C–E, F3),
or evaluation sampling (F1(b)). Follow the outputs for your assigned module.
"""
import sys
import sympy as sp
R = sp.Rational
Z = sp.Matrix([R(2,3),R(5,6)])
P_OBJECTIVE = (R(1,3),)*3
Q_ACQUISITION = (R(1,15),R(1,3),R(3,5))
INTEGRAND = (1,0,-1)
Q_MISSING = (0, R(1, 3), R(2, 3))  # Normalized; fails the target support check.
# F1(b): both, A only, B only, neither.
EVALUATION_PAIRED_COUNTS = (8, 6, 2, 4)
EVALUATION_DOCUMENT_COUNTS = (
    (5, 0, 0, 0),
    (3, 2, 0, 0),
    (0, 4, 1, 0),
    (0, 0, 1, 4),
)

def part_a():
    """Preserve allocation mathematics. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part A")

def part_b():
    """Separate mixtures. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part B")

def part_c():
    """Importance and variance. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part C")

def part_d():
    """Support and modified estimators. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part D")

def part_e():
    """Report distinct conditional means for the second contribution and the average."""
    raise NotImplementedError("Complete Part E")

def part_f():
    """F1(a) response-surface sensitivity, F1(b) evaluation, F2 lifetime, F3 design.

    For F1(b), reconstruct the item differences and document means
    from the supplied counts. Keep their independent units separate, explain
    the exact test's scope, and calculate the prespecified family threshold.
    """
    raise NotImplementedError("Complete Part F")

if __name__ == "__main__":
    print("Python",sys.version.split()[0],"SymPy",sp.__version__)
    print("Inputs loaded for the three printed modules.")
