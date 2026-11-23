"""Instructor solution for the Chapter 2 finite symbolic prediction head.

Symbolic differentiation and independently assembled matrix rules are kept
separate. Numerical loss evaluation below does not call either gradient path.
This is a correctness demonstration, not a performance measurement.
"""

import math

import sympy as sp


def rvec(matrix):
    """List matrix entries row by row as a column."""
    return sp.Matrix([matrix[i, j] for i in range(matrix.rows)
                      for j in range(matrix.cols)])


def symbolic_model():
    xs = sp.symbols("x00 x01 x10 x11", real=True)
    ws = sp.symbols("w00 w01 w02 w10 w11 w12", real=True)
    bs = sp.symbols("b0 b1 b2", real=True)
    theta = (*xs, *ws, *bs)
    X, W, b = sp.Matrix(2, 2, xs), sp.Matrix(2, 3, ws), sp.Matrix(bs)
    T = sp.Matrix([[1, 0, 0], [0, 0, 1]])
    Z = X * W + sp.ones(2, 1) * b.T
    denominators = [sum(sp.exp(Z[i, j]) for j in range(3)) for i in range(2)]
    P = sp.Matrix(2, 3, lambda i, j: sp.exp(Z[i, j]) / denominators[i])
    L = (sp.log(denominators[0]) - Z[0, 0]
         + sp.log(denominators[1]) - Z[1, 2]) / 2

    # Path B1: differentiate only the scalar expression.
    gradient = sp.Matrix([sp.diff(L, variable) for variable in theta])
    # Path B2: assemble the matrix rules without consulting B1.
    G = (P - T) / 2
    barX, barW, barb = G * W.T, X.T * G, G.T * sp.ones(2, 1)
    assembled = rvec(barX).col_join(rvec(barW)).col_join(barb)
    return dict(theta=theta, X=X, W=W, b=b, T=T, Z=Z, P=P, L=L,
                gradient=gradient, G=G, barX=barX, barW=barW, barb=barb,
                assembled=assembled)


def fixture_values(zero_weights=False):
    """Required exact log-valued fixture; entries remain independent until substitution."""
    weights = [sp.log(2), -sp.log(3), 0, 0, sp.log(3), 0]
    if zero_weights:
        weights = [0] * 6
    return tuple(map(sp.sympify, [1, 1, 0, 1, *weights, 0, 0, 0]))


def historical_fixture_values():
    """Former rational fixture, retained only for the optional precision experiment."""
    return tuple(map(sp.sympify, [1, -1, 2, 1, sp.Rational(1, 2), 0,
                                  sp.Rational(-1, 2), sp.Rational(-1, 4),
                                  sp.Rational(1, 4), 0, 0, sp.Rational(1, 4),
                                  sp.Rational(-1, 4)]))


def shared_model():
    """Independent target q preserves the stopped path until after differentiation."""
    entries = sp.symbols("a00 a01 a10 a11", real=True)
    A = sp.Matrix(2, 2, entries)
    a = sp.Matrix(sp.symbols("a0:2", real=True))
    b = sp.Matrix(sp.symbols("c0:2", real=True))
    q = sp.Matrix(sp.symbols("q0:2", real=True))
    r = A * a - A * b
    L = (r.T * r)[0] / 2
    residual = A * a - q
    frozen = (residual.T * residual)[0] / 2
    full_gradient = sp.Matrix(2, 2, [sp.diff(L, x) for x in entries])
    frozen_gradient = sp.Matrix(2, 2, [sp.diff(frozen, x) for x in entries])
    target = dict(zip(q, A * b))
    stopped = frozen_gradient.subs(target, simultaneous=True)
    return dict(A=A, entries=entries, a=a, b=b, q=q, r=r, L=L, frozen=frozen,
                full_gradient=full_gradient, frozen_gradient=frozen_gradient,
                stopped=stopped, target=target)


def scalar_loss(values, arithmetic=math, common_shift=0):
    """Independent scalar implementation, using shifted log-sum-exp.

    arithmetic is math for Python floats or mpmath for arbitrary precision.
    Targets are first class in row 0 and third class in row 1.
    """
    row_losses = []
    for i, target in enumerate((0, 2)):
        logits = [sum(values[2 * i + a] * values[4 + 3 * a + j]
                      for a in range(2)) + values[10 + j] + common_shift
                  for j in range(3)]
        maximum = max(logits)
        logsum = arithmetic.log(sum(arithmetic.exp(z - maximum) for z in logits))
        row_losses.append(logsum - (logits[target] - maximum))
    return sum(row_losses) / 2


def directions(arithmetic=math):
    first = [j / arithmetic.sqrt(819) for j in range(1, 14)]
    second = [((-1) ** j) / arithmetic.sqrt(13) for j in range(13)]
    third = [0] * 13
    third[11] = 1  # e_12 uses one-based mathematical indexing: middle bias b1.
    return (first, second, third)


def central_difference(values, direction, step, arithmetic=math):
    plus = [x + step * v for x, v in zip(values, direction)]
    minus = [x - step * v for x, v in zip(values, direction)]
    return (scalar_loss(plus, arithmetic) - scalar_loss(minus, arithmetic)) / (2 * step)
