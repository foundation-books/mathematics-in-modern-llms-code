"""Exact instructor calculations for Appendix A's three-part investigation."""

from itertools import combinations
from math import comb
import platform

import sympy as sp

from homework_starter import DOCUMENT_COUNTS, FACTORIAL, PAIRED_COUNTS, POINTS, STRATA


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Counts must be nonnegative integers")
    return value


def pairs_from_counts(counts):
    if len(counts) != 4:
        raise ValueError("Supply four counts: both, A only, B only, neither")
    return [pair for pair, number in zip(((1, 1), (1, 0), (0, 1), (0, 0)), counts)
            for _ in range(count(number))]


def sample_statistics(values):
    if len(values) < 2:
        raise ValueError("Sample variance requires at least two observations")
    values = list(map(sp.Rational, values))
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean)**2 for value in values) / (n - 1)
    return mean, variance, variance / n


def paired_statistics(counts=PAIRED_COUNTS):
    pairs = pairs_from_counts(counts)
    a, b = zip(*pairs) if pairs else ((), ())
    differences = [x - y for x, y in pairs]
    mean, variance, se2 = sample_statistics(differences)
    mean_a, variance_a, _ = sample_statistics(a)
    mean_b, variance_b, _ = sample_statistics(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in pairs) / (len(pairs) - 1)
    return dict(n=len(pairs), mean_a=mean_a, mean_b=mean_b, mean=mean,
                sum_squares=sum(d*d for d in differences), variance=variance,
                se2=se2, se=sp.sqrt(se2), covariance=covariance,
                variance_a=variance_a, variance_b=variance_b)


def exact_pvalue(n10, n01):
    """Twice the smaller inclusive Binomial(m, 1/2) tail, capped at one."""
    m = count(n10) + count(n01)
    if m == 0:
        return sp.Integer(1)
    tail = sum(comb(m, j) for j in range(min(n10, n01) + 1))
    return min(sp.Integer(1), sp.Rational(2*tail, 2**m))


def document_statistics(rows=DOCUMENT_COUNTS):
    means = []
    for row in rows:
        pairs = pairs_from_counts(row)
        if not pairs:
            raise ValueError("A document must contain at least one item")
        means.append(sp.Rational(sum(a - b for a, b in pairs), len(pairs)))
    return means, sample_statistics(means)


def factorial_effects(scores=FACTORIAL):
    x0 = scores[1, 0] - scores[0, 0]
    x1 = scores[1, 1] - scores[0, 1]
    y0 = scores[0, 1] - scores[0, 0]
    y1 = scores[1, 1] - scores[1, 0]
    return dict(x0=x0, x1=x1, y0=y0, y1=y1,
                main_x=(x0 + x1)/2, main_y=(y0 + y1)/2, interaction=x1 - x0)


def mixture_scores():
    w = sp.Symbol("w", real=True)
    scores = {name: sp.expand(w*easy + (1 - w)*hard)
              for name, (easy, hard) in STRATA.items()}
    return w, scores, sp.expand(scores["A"] - scores["B"])


def dominates(a, b):
    qa, ca = a
    qb, cb = b
    return qa >= qb and ca <= cb and (qa > qb or ca < cb)


def pareto_set(points=POINTS):
    return {name for name, point in points.items()
            if not any(dominates(other, point) for other in points.values())}


def utility_regions(points=POINTS):
    weight = sp.Symbol("lambda", real=True)
    return {name: sp.reduce_inequalities(
                [weight >= 0] + [quality - weight*cost >= q - weight*c
                                 for q, c in points.values()], weight).as_set()
            for name, (quality, cost) in points.items()}


def utility_winners(weight, points=POINTS):
    if weight < 0:
        raise ValueError("Use a nonnegative cost weight")
    utilities = {name: q - weight*c for name, (q, c) in points.items()}
    best = max(utilities.values())
    return {name for name, utility in utilities.items() if utility == best}


def cap_winners(cap, points=POINTS):
    feasible = {name: q for name, (q, c) in points.items() if c <= cap}
    if not feasible:
        return set()
    best = max(feasible.values())
    return {name for name, quality in feasible.items() if quality == best}


def main():
    if not __debug__:
        raise RuntimeError("Run without -O: exact checks must remain enabled")
    print(f"Python {platform.python_version()}; SymPy {sp.__version__}")
    r = sp.Rational
    paired = paired_statistics()
    assert paired["n"] == 20
    assert (paired["mean_a"], paired["mean_b"], paired["mean"]) == (r(7, 10), r(1, 2), r(1, 5))
    assert (paired["sum_squares"], paired["variance"], paired["se2"]) == (8, r(36, 95), r(9, 475))
    assert paired["covariance"] == r(1, 19)
    assert paired["variance_a"] + paired["variance_b"] - 2*paired["covariance"] == paired["variance"]
    assert exact_pvalue(6, 2) == r(37, 128)
    assert exact_pvalue(0, 0) == exact_pvalue(4, 4) == 1
    means, document = document_statistics()
    assert means == [0, r(2, 5), r(3, 5), -r(1, 5)]
    assert document == (r(1, 5), r(2, 15), r(1, 30))
    k = len(list(combinations(range(5), 2)))
    assert k == 10 and r(1, 20)/k == r(1, 200)
    print("Part I:", paired)
    print("Exact p-value:", exact_pvalue(6, 2), "document means:", means, "document mean/variance/SE²:", document)
    print("Bonferroni:", r(1, 20)/k, "independent-null any rejection:", 1 - r(19, 20)**k)
    symbols = {(x, y): sp.Symbol(f"mu_{x}{y}") for x in (0, 1) for y in (0, 1)}
    symbolic_effects = factorial_effects(symbols)
    substitution = {symbols[cell]: score for cell, score in FACTORIAL.items()}
    effects = {name: expr.subs(substitution) for name, expr in symbolic_effects.items()}
    assert effects == factorial_effects()
    assert tuple(effects.values()) == (r(1, 10), r(3, 20), r(1, 20), r(1, 10), r(1, 8), r(3, 40), r(1, 20))
    w, scores, difference = mixture_scores()
    assert difference == (3*w - 2)/10
    assert sp.solve(difference, w) == [r(2, 3)]
    print("Part II symbolic:", symbolic_effects)
    print("Part II values:", effects, "mixture:", scores, "difference:", difference)
    for value in (r(1, 2), r(4, 5)):
        print("Weight", value, {name: expr.subs(w, value) for name, expr in scores.items()})
    regions = utility_regions()
    assert pareto_set() == {"A", "B", "D", "E"}
    assert regions == {"A": sp.Interval(r(1, 80), r(1, 40)), "B": sp.Interval(r(1, 40), sp.oo),
                       "C": sp.EmptySet, "D": sp.Interval(0, r(1, 80)), "E": sp.EmptySet}
    assert utility_winners(r(1, 80)) == {"D", "A"}
    assert utility_winners(r(1, 40)) == {"A", "B"}
    assert cap_winners(3) == {"E"}
    print("Part III: Pareto set:", sorted(pareto_set()), "maximizing weight regions:", regions)
    print("E versus B needs lambda <= 1/50; E versus A needs lambda >= 3/100.")
    print("Cost cap three selects:", sorted(cap_winners(3)))


if __name__ == "__main__":
    main()
