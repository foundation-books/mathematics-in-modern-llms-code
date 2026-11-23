"""Speculative decoding seed for the sampler route (A1–A6).

Use the chapter's separate Boolean fixtures for the language route (B–F).
"""
from __future__ import annotations
from typing import Sequence
import sys
import sympy as sp

R = sp.Rational


def categorical(values: Sequence[sp.Expr] | sp.MatrixBase) -> sp.Matrix:
    """Validate a nonempty exact rational categorical column vector."""
    v = sp.Matrix(values)
    if v.cols != 1 or v.rows == 0:
        raise ValueError("Expected a nonempty probability column vector.")
    if any(x.is_Rational is not True or x < 0 for x in v):
        raise ValueError("Use exact nonnegative rational probabilities.")
    if sum(v) != 1:
        raise ValueError("Probabilities must sum to one exactly.")
    return v


def accepted_mass(p: sp.Matrix, q: sp.Matrix) -> sp.Matrix:
    p, q = categorical(p), categorical(q)
    if p.shape != q.shape:
        raise ValueError("Target and proposal vocabularies must agree.")
    return sp.Matrix([min(px, qx) for px, qx in zip(p, q)])


o_star = sp.Matrix([R(2, 3), R(5, 6)])
p = categorical([R(1, 2), R(1, 3), R(1, 6)])
q = categorical([R(1, 3), R(1, 2), R(1, 6)])
P = sp.Matrix([
    [R(1, 2), R(1, 4), R(1, 4)],
    [R(1, 4), R(1, 2), R(1, 4)],
    [R(1, 4), R(1, 4), R(1, 2)],
])
Q = sp.Matrix([
    [R(1, 4), R(1, 2), R(1, 4)],
    [R(1, 2), R(1, 3), R(1, 6)],
    [R(1, 4), R(1, 4), R(1, 2)],
])
Q_after_C_changed = sp.Matrix([[1, 0, 0]])  # A4: replace only Q's C row.

m = accepted_mass(p, q)
assert sum(m) == R(5, 6)
print("Python:", sys.version.split()[0])
print("SymPy:", sp.__version__)
print("Accepted mass:", list(m))
