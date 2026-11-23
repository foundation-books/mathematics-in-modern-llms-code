"""Data and symbols for the Chapter 4 homework."""

import sys

import sympy as sp

tau = sp.symbols("tau", positive=True)
s = sp.Matrix(sp.symbols("s1 s2 s3", real=True))
q_star = sp.Matrix([sp.sqrt(2), 0])
K_star = sp.Matrix([[0, 0], [sp.log(2), 0], [sp.log(3), 0]])
V_star = sp.Matrix([[1, 0], [0, 1], [1, 1]])
# Local aliases for the fixed inputs; s remains an independent score column.
q, K, V = q_star, K_star, V_star
target = sp.Matrix([1, 0])
half = sp.Rational(1, 2)

assert s.shape == (3, 1)
assert q.shape == (2, 1)
assert K.shape == (3, 2)
assert V.shape == (3, 2)

if __name__ == "__main__":
    print("Python:", sys.version.split()[0])
    print("SymPy:", sp.__version__)
    print("Homework data and symbolic variables are ready.")
