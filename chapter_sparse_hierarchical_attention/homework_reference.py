"""Exact Chapter 7 homework reference; attempt the printed A–F tasks first.

Adapted from doc/ch07/chapter_07_homework_solution.py and its check helpers.
Written derivations remain in the book and the separate instructor material.
"""
from itertools import combinations, product
import sys

import sympy as sp

from reference import (
    block_route, block_statistics, fixture, pair_counts, policy_supports,
    pooled_attention, restricted_attention, top_k,
)

Q, K, V = fixture()
q = Q.row(0).T


def zero(expression):
    if isinstance(expression, sp.MatrixBase):
        return all(sp.simplify(x) == 0 for x in expression)
    return sp.simplify(expression) == 0


def diagnostics(support):
    a, dense = restricted_attention(8, q, K, V, range(1, 9))
    selected, output = restricted_attention(8, q, K, V, support)
    mass = sum((a[j - 1] for j in support), sp.S.Zero)
    delta = 1 - mass
    l1 = sum(abs(x) for x in a - selected)
    error = sp.simplify((dense - output).norm())
    return mass, delta, l1, error, output




def main():
    Q, K, V = fixture()
    q = Q.row(0).T
    print(f"Python {sys.version.split()[0]}; SymPy {sp.__version__}")
    print("A: logits", (K * q / sp.sqrt(2)).T)
    for name in ("dense", "local"):
        a, output = restricted_attention(8, q, K, V, policy_supports()[name])
        assert sum(a) == 1
        print(name, "p=", a.T, "o=", output.T)
    for malformed in ((), (1, 1), (9,), (0,), (True,), (1.0,)):
        try:
            restricted_attention(8, q, K, V, malformed)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("malformed support accepted")
    print("B: prefix table (t, selected blocks, support, output)")
    for t in range(1, 9):
        chosen, support, _ = block_route(t, q, K)
        _, out = restricted_attention(t, q, K, V, support)
        prefix = block_route(t, q, K[:t, :])
        assert prefix[:2] == (chosen, support)
        assert zero(out - restricted_attention(t, q, K[:t, :], V[:t, :], prefix[1])[1])
        for future in range(t + 1, 9):
            changed = K.copy()
            changed[future - 1, 0] = sp.log(128)
            changed_route = block_route(t, q, changed)
            assert changed_route[:2] == (chosen, support)
            assert zero(out - restricted_attention(t, q, changed, V, support)[1])
        print(t, chosen, support, tuple(out))
    changed = K.copy()
    changed[7, 0] = sp.log(128)
    faulty = [block_route(7, q, keys, faulty_current=True) for keys in (K, changed)]
    assert faulty[0][1] != faulty[1][1]
    print("Faulty contrast:", [entry[:2] for entry in faulty])
    print("C: policy (mass, delta, L1, Euclidean error, output)")
    for name, support in policy_supports().items():
        mass, delta, l1, error, output = diagnostics(support)
        assert l1 == 2 * delta
        assert sp.simplify(sp.sqrt(2) * delta - error).is_nonnegative
        print(name, mass, delta, l1, error, tuple(output))
    block_choices = [(diagnostics(block + (7, 8))[0], u) for u, block in
                     enumerate(((1, 2), (3, 4), (5, 6)), 1)]
    token_choices = [(diagnostics(ids + (7, 8))[0], ids)
                     for ids in combinations(range(1, 7), 2)]
    mass = max(item[0] for item in token_choices)
    best_ids = min(ids for m, ids in token_choices if m == mass)
    assert (max(block_choices), mass, best_ids) == ((sp.Rational(7, 12), 1), sp.Rational(17, 24), (1, 3))
    assert restricted_attention(8, q, K, sp.ones(8, 2), (7, 8))[1] == sp.ones(2, 1)
    print("Oracles:", max(block_choices), mass, best_ids)
    print("D: pooled variants")
    blocks = ((1, 2), (3, 4), (5, 6), (7, 8))
    for correction in (False, True):
        exponential_weights, output = pooled_attention(q, K, V, blocks[:3], (7, 8), correct_size=correction)
        print(correction, exponential_weights, tuple(output))
    masses, means = block_statistics(8, q, K, V, blocks)
    recombined = sum((m * value for m, value in zip(masses, means)), sp.zeros(2, 1)) / sum(masses)
    assert zero(recombined - restricted_attention(8, q, K, V, range(1, 9))[1])
    equalK = sp.Matrix([[sp.log(2), 0], [sp.log(2), 0], [sp.log(3), 0]])
    assert zero(pooled_attention(q, equalK, V[:3, :], ((1, 2),), (3,), correct_size=True)[1]
                - restricted_attention(3, q, equalK, V[:3, :], (1, 2, 3))[1])
    pool = top_k({1: 0, 2: 2, 3: 1}, 2)
    final = top_k({u: masses[u - 1] for u in pool}, 1)
    ceilings = (diagnostics((3, 4, 7, 8))[0], diagnostics((3, 4, 5, 6, 7, 8))[0])
    assert pool == (2, 3) and final == (2,)
    print("Z:", masses, "mu:", [tuple(value) for value in means], "ceilings:", ceilings)
    print("E: smooth derivative and strict-margin checks")
    theta = sp.symbols("theta", real=True)
    rho = sp.exp(theta) / (1 + sp.exp(theta))
    loss = -sp.Rational(3, 4) * sp.log(rho) - sp.Rational(1, 4) * sp.log(1 - rho)
    assert zero(sp.diff(loss, theta) - rho + sp.Rational(3, 4))
    print(sp.simplify(sp.diff(loss, theta)))
    epsilon = sp.log(sp.Rational(5, 4)) / 4
    for signs in product((-1, 1), repeat=3):
        exact = (sp.log(10), sp.log(8), sp.log(2))
        assert top_k({j: exact[j - 1] + signs[j - 1] * epsilon for j in (1, 2, 3)}, 1) == (1,)
    assert top_k({1: 0, 2: 2}, 1) != top_k({1: 1, 2: 1}, 1)
    print("F: counts and abstract costs")
    assert pair_counts(8, 2, 1) == (36, 15, 24, 12)
    b = sp.symbols("b", positive=True)
    alpha, beta, T = sp.symbols("alpha beta T", positive=True)
    kb = sp.symbols("k_b", nonnegative=True)
    cost = alpha * T**2 / b + beta * T * (kb + 1) * b
    optimum = sp.sqrt(alpha * T / (beta * (kb + 1)))
    assert zero(sp.diff(cost, b).subs(b, optimum))
    assert sp.diff(cost, b, 2).is_positive
    print("C'", sp.diff(cost, b), "C''", sp.diff(cost, b, 2), "b*", optimum)
    print(pair_counts(8, 2, 1), [cost.subs({alpha: 1, beta: 1, T: 1024, kb: 3, b: size})
                               for size in (8, 16, 32)])
    print("All A–F computational assertions passed; hand reasoning and memo are separate.")


if __name__ == "__main__":
    main()
