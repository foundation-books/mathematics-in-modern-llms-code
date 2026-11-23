"""Starter for Chapter 12; supplies data and tie semantics, not solutions."""
import sys
import sympy as sp
from fixture import (
    ALPHA, BIAS_DEMO, BIAS_ZERO, D, EXPERTS, LATENT_FACTORS, MARGINS,
    S, SHARED, TAU, TIED_MARGINS, U, X, exact_topk,
)

if __name__ == "__main__":
    print("Python", sys.version.split()[0], "SymPy", sp.__version__)
    print("Inputs:", X.shape, "scores:", S.shape)
    print("Tie check:", exact_topk([sp.Rational(1, 2)] * 3, 2))
    print("Implement parts A–F from the printed assignment using these data.")
