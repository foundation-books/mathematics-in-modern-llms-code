"""Exact teaching data printed in Chapter 6; these are not trained weights."""

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class Fixture:
    X: sp.Matrix
    D: sp.Matrix
    WQ: tuple
    UK: tuple
    UV: tuple
    WO_col: tuple


def fixture():
    X = sp.Matrix([[0, sp.log(2), sp.log(3)], [1, 0, 1],
                   [0, 1, 1], [1, 1, 1]])
    D = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    queries = tuple(sp.Matrix(q) for q in [(1, 0), (0, 1), (1, 1), (1, -1)])
    e4 = sp.Matrix([0, 0, 0, 1])
    WQ = tuple(q * e4.T for q in queries)
    UK = tuple(sp.Matrix([[sp.sqrt(2), 0, 0], [0, a, b]])
               for a, b in [(1, -1), (1, 1), (-1, 1), (2, -1)])
    UV = tuple(sp.Matrix(v) for v in [
        [[0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0]],
        [[0, 1, 1], [0, 1, -1]], [[0, 2, 0], [0, 0, 1]],
    ])
    WO1_col = sp.Matrix([[1, 0], [0, 1], [0, 0], [0, 0]])
    WO2_col = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
    return Fixture(X, D, WQ, UK, UV, (WO1_col, WO2_col, WO1_col / 2, WO2_col / 2))


def softmax_exact(scores):
    scores = sp.Matrix(scores)
    if scores.cols != 1 or scores.rows == 0:
        raise ValueError("scores must be a nonempty column vector")
    terms = scores.applyfunc(lambda z: sp.exp(sp.simplify(z)))
    return terms / sum(terms)


def assert_zero(matrix):
    residual = sp.Matrix(matrix).applyfunc(sp.simplify)
    if any(entry != 0 for entry in residual):
        raise AssertionError(f"Nonzero or unresolved residual: {residual}")


if __name__ == "__main__":
    import sys
    data = fixture()
    print("Python:", sys.version.split()[0])
    print("SymPy:", sp.__version__)
    print("Input/latent shapes:", data.X.shape, (data.D * data.X).shape)
