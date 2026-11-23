"""Inputs for Chapter 5's four-part efficient attention SymPy homework."""
from dataclasses import dataclass
import sympy as sp

scores = (sp.Integer(0), sp.log(2), sp.log(3))
values = sp.Matrix([[1, 0], [0, 1], [1, 1]])
q = sp.Matrix([1, 0])
keys = sp.Matrix([[0, 0], [sp.sqrt(2)*sp.log(2), 0],
                  [sp.sqrt(2)*sp.log(3), 0]])
queries = sp.Matrix.vstack(q.T, q.T, q.T)
allowed = ((0,), (0, 1), (0, 1, 2))
upstream = sp.Matrix([1, 2])
partitions = (((0,), (1, 2)), ((0, 1), (2,)))
order = (2, 0, 1)
changed_values = sp.Matrix([[1, 0], [0, 1], [7, -4]])

n, dk, dv = 8, 2, 2
io_bytes, acc_bytes, fast_memory_bytes = 2, 4, 96
tiles = ((2, 2), (4, 2))
contraction_flops = 12 * 10**9
traffic_bytes = (12 * 10**8, 3 * 10**8)
peak_flops_per_second = 120 * 10**12
peak_bytes_per_second = 3 * 10**12

reference_digits, working_digits = 70, (8, 12)
atol, rtol = sp.Rational(1, 10**7), sp.Rational(1, 10**5)
score_shift, value_scale = 1000, 10**6
cancellation_scores = (0, 0, 0)
cancellation_values = sp.Matrix([[10**12, 1], [-10**12, 1], [1, -2]])
cancellation_orders = ((0, 1, 2), (0, 2, 1))


@dataclass(frozen=True)
class Summary:
    maximum: sp.Expr
    denominator: sp.Expr
    numerator: sp.ImmutableMatrix


def summarize(ids, row_scores, row_values):
    raise NotImplementedError("Part B: canonical summary; None for empty.")


def merge(a, b):
    raise NotImplementedError("Part B: common-scale merge and tagged identities.")


def finalize(state):
    raise NotImplementedError("Part B: normalize or reject the empty final row.")


if __name__ == '__main__':
    print({'Q': queries.shape, 'K': keys.shape, 'V': values.shape,
           'allowed_zero_based': allowed, 'parts': ('A', 'B', 'C', 'D'),
           'tiles': tiles, 'working_decimal_digits': working_digits})
