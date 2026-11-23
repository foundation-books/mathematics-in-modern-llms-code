"""Exact layer-local teaching data; this is not a preceding decoder output."""
from __future__ import annotations

from collections.abc import Sequence
import sympy as sp

R = sp.Rational
X = sp.Matrix([[1, 0], [0, 1], [R(2, 3), R(5, 6)]])
S = sp.Matrix([[9, 8, 1], [8, 7, 2], [7, 3, 6]]) / 10
BIAS_ZERO = sp.zeros(1, 3)
BIAS_DEMO = sp.Matrix([[R(-3, 10), 0, R(2, 5)]])
EXPERTS = (sp.eye(2), sp.diag(2, 1), sp.Matrix([[0, 1], [1, 0]]))
SHARED = sp.eye(2) / 2
ALPHA = sp.Matrix([R(11, 20), R(11, 20), R(7, 20)])
TAU = sp.Matrix([[R(3, 10), 0, R(-2, 5)]])
MARGINS = (R(4, 5), R(1, 2), R(1, 10), R(-2, 5))
TIED_MARGINS = (R(4, 5), R(1, 2), R(1, 2), R(-2, 5))
D = sp.Matrix([1, 1])
U = sp.Matrix([[1, -1]])
LATENT_FACTORS = (sp.Integer(1), sp.Integer(2), sp.Integer(-1))


def exact_topk(scores: Sequence[sp.Expr], k: int) -> tuple[int, ...]:
    """Zero-based IDs; descending scores, then ascending ID on ties."""
    values = tuple(sp.sympify(v) for v in scores)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(values):
        raise ValueError("k must be an integer between 1 and the expert count")
    if any(v.is_Rational is not True for v in values):
        raise TypeError("concrete integer/rational scores are required")
    return tuple(sorted(range(len(values)), key=lambda j: (-values[j], j))[:k])
