"""Exact teaching fixtures; no model or training measurements."""
import sys
import sympy as sp

Q = sp.Rational
X0 = sp.Matrix([Q(2, 3), Q(5, 6)])
A = sp.Matrix([[0, Q(1, 2)], [0, 0]])
B = sp.Matrix([[0, 0], [-Q(1, 3), 0]])
K0 = sp.Matrix([[1, 2], [3, 4]])


def check_fixtures():
    r0 = A * X0
    x1 = X0 + r0
    r1 = B * x1
    x2 = x1 + r1
    assert r0 == sp.Matrix([Q(5, 12), 0])
    assert x1 == sp.Matrix([Q(13, 12), Q(5, 6)])
    assert r1 == sp.Matrix([0, -Q(13, 36)])
    assert x2 == sp.Matrix([Q(13, 12), Q(17, 36)])
    assert (sp.eye(2) + B) * (sp.eye(2) + A) == sp.Matrix(
        [[1, Q(1, 2)], [-Q(1, 3), Q(5, 6)]])
    v2 = B * (X0 + r0) / 2
    assert (X0 + r0 + v2) / 3 == sp.Matrix([Q(13, 36), Q(47, 216)])
    assert (X0 + r0 + v2) / 2 == sp.Matrix([Q(13, 24), Q(47, 144)])
    return x1, x2


if __name__ == "__main__":
    check_fixtures()
    print(f"Fixture checks passed: Python {sys.version.split()[0]}, SymPy {sp.__version__}")
