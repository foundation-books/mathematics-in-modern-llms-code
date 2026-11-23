"""Exact teaching computations; no training or production performance claims."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import sys
import sympy as sp

from fixture import (
    ALPHA, BIAS_DEMO, BIAS_ZERO, D, EXPERTS, MARGINS, R, S, SHARED,
    TAU, TIED_MARGINS, U, X, exact_topk,
)


@dataclass(frozen=True)
class Route:
    origin: int
    expert: int
    slot: int
    weight: sp.Expr
    payload: sp.ImmutableMatrix


@dataclass
class Layer:
    selected: tuple[tuple[int, ...], ...]
    weights: sp.Matrix
    incidence: sp.Matrix
    output: sp.Matrix
    records: tuple[Route, ...]

    @property
    def loads(self) -> tuple[sp.Expr, ...]:
        return tuple(sum(self.incidence[:, j]) for j in range(self.incidence.cols))


def combine(records, inputs, experts, shared):
    """Combine given records; semantic record preservation is a caller invariant."""
    result = inputs * shared
    for record in records:
        contribution = record.weight * record.payload * experts[record.expert]
        result[record.origin, :] = result[record.origin, :] + contribution
    return sp.Matrix(result)


def layer(inputs=X, scores=S, bias=BIAS_ZERO, experts=EXPERTS, shared=SHARED, k=2):
    """Linear fixture layer, using positive rational affinities and row vectors."""
    n, d = inputs.shape
    e = scores.cols
    if scores.rows != n or bias.shape != (1, e) or len(experts) != e:
        raise ValueError("incompatible routing shapes")
    if shared.shape != (d, d) or any(f.shape != (d, d) for f in experts):
        raise ValueError("experts must preserve model width")
    if any(v.is_Rational is not True or v <= 0 for v in scores):
        raise ValueError("this fixture requires positive rational affinities")
    weights, incidence = sp.zeros(n, e), sp.zeros(n, e)
    selected, records = [], []
    for i in range(n):
        ids = exact_topk([scores[i, j] + bias[j] for j in range(e)], k)
        selected.append(ids)
        denominator = sum(scores[i, j] for j in ids)
        for slot, j in enumerate(ids):
            weight = scores[i, j] / denominator
            weights[i, j], incidence[i, j] = weight, 1
            records.append(Route(i, j, slot, weight, sp.ImmutableMatrix(inputs[i, :])))
    records = tuple(records)
    return Layer(tuple(selected), weights, incidence,
                 combine(records, inputs, experts, shared), records)


def imbalance(loads):
    q = sum(loads) / sp.Integer(len(loads))
    return sp.simplify(sum((n / q - 1) ** 2 for n in loads) / len(loads))


def overflow(loads, capacity):
    return sum(max(0, n - capacity) for n in loads)


def sign_step(bias, loads, eta):
    q = sum(loads) / sp.Integer(len(loads))
    return bias + sp.Matrix([[eta * sp.sign(q - n) for n in loads]])


def assignment_matrix(selected, e):
    a = sp.zeros(len(selected), e)
    for i, ids in enumerate(selected):
        for j in ids:
            a[i, j] = 1
    return a


def balanced_assignments(scores=S, k=2):
    n, e = scores.shape
    q = R(n * k, e)
    if not q.is_Integer:
        raise ValueError("equal integer column quotas are required")
    candidates, feasible = 0, []
    for selected in product(tuple(combinations(range(e), k)), repeat=n):
        candidates += 1
        a = assignment_matrix(selected, e)
        if all(sum(a[:, j]) == q for j in range(e)):
            score = sum(a[i, j] * scores[i, j] for i in range(n) for j in range(e))
            feasible.append((selected, score))
    return candidates, feasible


def dual_bound(scores=S, alpha=ALPHA, tau=TAU, k=2):
    n, e = scores.shape
    q = R(n * k, e)
    return (k * sum(alpha) + q * sum(tau)
            + sum(max(0, scores[i, j] - alpha[i] - tau[j])
                  for i in range(n) for j in range(e)))


def threshold_interval(margins, q):
    """Exact minimizers, including endpoint quotas; no interpolated quantiles."""
    values = tuple(sp.sympify(v) for v in margins)
    if not values or any(v.is_Rational is not True for v in values):
        raise TypeError("nonempty concrete rational margins required")
    if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= len(values):
        raise ValueError("q must be an integer from zero through the sample count")
    ordered = sorted(values, reverse=True)
    if q == 0:
        return ordered[0], sp.oo
    if q == len(values):
        return -sp.oo, ordered[-1]
    return ordered[q], ordered[q - 1]


def hinge(margins, q, t):
    return q * t + sum(sp.Max(0, m - t) for m in margins)


def quantile_candidate(scores, incoming, k):
    """Exact integer-quota surrogate of the report's ordered next-step rule.

    Not its histogram implementation or a guarantee of reconstructed balance.
    Returns biased row cutoffs, raw margins, and centered candidate. k=E has
    no balancing cutoff and is handled separately.
    """
    n, e = scores.shape
    if incoming.shape != (1, e):
        raise ValueError("incoming bias must have one value per expert")
    if k == e:
        return None, None, incoming - sp.ones(1, e) * (sum(incoming) / e)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k < e:
        raise ValueError("require 1 <= k <= E")
    q = R(n * k, e)
    if not q.is_Integer or not 1 <= q < n:
        raise ValueError("exact surrogate requires an interior integer target")
    alpha = sp.Matrix([
        scores[i, ids[-1]] + incoming[ids[-1]]
        for i in range(n)
        for ids in [exact_topk([scores[i, j] + incoming[j] for j in range(e)], k + 1)]
    ])
    margins = sp.Matrix(n, e, lambda i, j: scores[i, j] - alpha[i])
    tau = [threshold_interval(tuple(margins[:, j]), int(q))[0] for j in range(e)]
    candidate = -sp.Matrix([tau])
    candidate -= sp.ones(1, e) * (sum(candidate) / e)
    return alpha, margins, candidate


def wrong_biased_weights(scores, bias, ids):
    values = [scores[j] + bias[j] for j in ids]
    denominator = sum(values)
    return tuple(v / denominator for v in values)


def reference_checks():
    zero, demo = layer(), layer(bias=BIAS_DEMO)
    expected_zero = sp.Matrix([[R(67, 34), 0], [0, R(3, 2)], [R(14, 13), R(61, 52)]])
    expected_demo = expected_zero.copy()
    expected_demo[1, :] = sp.Matrix([[R(2, 9), R(23, 18)]])
    assert zero.output == expected_zero and demo.output == expected_demo
    assert zero.loads == (3, 2, 1) and demo.loads == (2, 2, 2)
    for result in (zero, demo):
        assert all(sum(result.weights[i, :]) == 1 for i in range(3))
        assert all(sum(result.incidence[i, :]) == 2 for i in range(3))
        assert sum(result.loads) == 6
    assert imbalance(zero.loads) == R(1, 6) and imbalance(demo.loads) == 0
    assert overflow(zero.loads, 2) == 1 and overflow(demo.loads, 2) == 0
    count, feasible = balanced_assignments()
    assert count == 27 and len(feasible) == 6
    assert max(score for _, score in feasible) == dual_bound() == R(39, 10)
    assert threshold_interval(MARGINS, 2) == (R(1, 10), R(1, 2))
    assert threshold_interval(TIED_MARGINS, 2) == (R(1, 2), R(1, 2))
    a, c = sp.symbols("a c", positive=True)
    jacobian = sp.Matrix([a / (a + c), c / (a + c)]).jacobian([a, c])
    assert sp.simplify(jacobian - sp.Matrix([[c, -a], [-c, a]]) / (a + c) ** 2) == sp.zeros(2)
    assert (R(3, 2) * D * U).rank() == 1
    assert (SHARED + R(3, 2) * D * U).det() == R(1, 4)
    assert (SHARED + R(3, 2) * D * U).rank() == 2
    # An affine sigmoid router realizes S exactly in logit space.
    augmented = X.row_join(sp.ones(3, 1))
    logits = S.applyfunc(lambda v: sp.log(v / (1 - v)))
    assert augmented.det() == R(-1, 2)
    router = augmented.inv() * logits
    assert sp.simplify(augmented * router - logits) == sp.zeros(3)
    print("Python", sys.version.split()[0], "SymPy", sp.__version__)
    print("Zero-bias output:", zero.output)
    print("Demo-bias output:", demo.output)
    print("All 27 assignments checked; six feasible; optimum/certificate:", dual_bound())
    print("One sign step:", sign_step(BIAS_ZERO, zero.loads, R(1, 10)))
    print("Exact cutoff-surrogate candidate from zero:", quantile_candidate(S, BIAS_ZERO, 2)[2])
    print("Exact fixture, derivative, threshold and geometry checks passed.")


if __name__ == "__main__":
    reference_checks()
