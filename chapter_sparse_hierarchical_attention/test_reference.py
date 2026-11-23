"""Clean-run exact checks for Chapter 7; finite checks support general proofs."""

from itertools import combinations, product
import sys
import unittest

import sympy as sp

from reference import (
    block_route, block_statistics, fixture, packed_route, pair_counts,
    policy_supports, pooled_attention, restricted_attention, top_k,
)
from homework_reference import diagnostics, zero

Q, K, V = fixture()
q = Q.row(0).T
BLOCKS = ((1, 2), (3, 4), (5, 6), (7, 8))
R = sp.Rational


class Chapter7Checks(unittest.TestCase):
    def test_input_shapes_scores_and_failures(self):
        self.assertEqual(Q.shape, (8, 2))
        self.assertEqual(K.shape, (8, 2))
        self.assertEqual(V.shape, (8, 2))
        for t in range(1, 9):
            for j in range(1, t + 1):
                self.assertTrue(zero((Q.row(t - 1) * K.row(j - 1).T)[0]
                                     / sp.sqrt(2) - K[j - 1, 0]))
        for support in ((), (1, 1), (0,), (-1,), (4,), (9,), (True,), (1.0,)):
            with self.subTest(support=support):
                with self.assertRaises((TypeError, ValueError)):
                    restricted_attention(3, q, K, V, support)
        for t in (0, 9, True, 3.0):
            with self.assertRaises((TypeError, ValueError)):
                restricted_attention(t, q, K, V, (1,))
        for bad_q, bad_K, bad_V in ((q.T, K, V), (q, K[:, :1], V),
                                    (q, K, V[:7, :])):
            with self.assertRaises(ValueError):
                restricted_attention(3, bad_q, bad_K, bad_V, (1,))
        bad_K = K.copy()
        bad_K[0, 0] = sp.oo
        with self.assertRaises(ValueError):
            restricted_attention(3, q, bad_K, V, (1,))
        for exponential_weights in ((), (0,), (-1,), (sp.oo,)):
            with self.assertRaises(ValueError):
                fixture(exponential_weights)

    def test_outputs_and_conditioning_all_nonempty_fixture_supports(self):
        # Exhaustive declared finite domain: all 255 nonempty supports at t=8.
        a, dense = restricted_attention(8, q, K, V, range(1, 9))
        count = 0
        for size in range(1, 9):
            for ids in combinations(range(1, 9), size):
                selected, output = restricted_attention(8, q, K, V, ids)
                mass = sum(a[j - 1] for j in ids)
                expected = sp.Matrix([a[j - 1] / mass if j in ids else 0
                                      for j in range(1, 9)])
                self.assertTrue(zero(selected - expected))
                self.assertEqual(sum(selected), 1)
                self.assertEqual(sum(abs(x) for x in a - selected), 2 * (1 - mass))
                if mass == 1:
                    self.assertTrue(zero(output - dense))
                count += 1
        self.assertEqual(count, 255)

    def test_five_policies_and_bounds(self):
        expected = {
            "dense": (1, (R(13, 24), R(1, 2))),
            "local": (R(1, 6), (1, 1)),
            "mean-key block": (R(1, 2), (R(1, 3), 1)),
            "block mass oracle": (R(7, 12), (R(13, 14), R(2, 7))),
            "token mass oracle": (R(17, 24), (R(13, 17), R(8, 17))),
        }
        M = max(V.row(j).norm() for j in range(8))
        D = max((V.row(i) - V.row(j)).norm() for i in range(8) for j in range(8))
        self.assertEqual(M, sp.sqrt(2))
        self.assertEqual(D, sp.sqrt(2))
        target = set(policy_supports()["token mass oracle"])
        expected_recalls = {
            "dense": 1, "local": R(1, 2), "mean-key block": R(3, 4),
            "block mass oracle": R(3, 4), "token mass oracle": 1,
        }
        for name, ids in policy_supports().items():
            mass, delta, l1, error, output = diagnostics(ids)
            self.assertEqual(mass, expected[name][0])
            self.assertTrue(zero(output - sp.Matrix(expected[name][1])))
            self.assertEqual(l1, 2 * delta)
            self.assertTrue(sp.simplify(D * delta - error).is_nonnegative)
            self.assertTrue(sp.simplify(2 * M * delta - error).is_nonnegative)
            recall = R(len(set(ids) & target), len(target))
            self.assertEqual(recall, expected_recalls[name])
        self.assertEqual(diagnostics((3, 4, 7, 8))[3], R(13, 24))
        a, dense = restricted_attention(8, q, K, V, range(1, 9))
        ps, selected = restricted_attention(8, q, K, V, (3, 4, 7, 8))
        omitted = sum((a[j - 1] * V.row(j - 1).T for j in (1, 2, 5, 6)),
                      sp.zeros(2, 1)) / R(1, 2)
        self.assertTrue(zero(dense - selected / 2 - omitted / 2))

    def test_router_all_prefixes_and_each_future_suffix_position(self):
        expected_supports = ((1,), (1, 2), (1, 2, 3), (1, 2, 3, 4),
                             (3, 4, 5), (3, 4, 5, 6), (3, 4, 7), (3, 4, 7, 8))
        full = []
        for t in range(1, 9):
            ids, support, scores = block_route(t, q, K)
            self.assertEqual(support, expected_supports[t - 1])
            a, out = restricted_attention(t, q, K, V, support)
            Qp, Kp, Vp = fixture((9, 1, 4, 4, 1, 1, 2, 2)[:t])
            prefix_ids, prefix_support, prefix_scores = block_route(t, Qp.row(t - 1).T, Kp)
            pp, prefix_out = restricted_attention(t, q, Kp, Vp, prefix_support)
            self.assertEqual((ids, support, scores), (prefix_ids, prefix_support, prefix_scores))
            self.assertTrue(zero(a - pp) and zero(out - prefix_out))
            full.append((ids, support, out))
            if t < 8:
                for future in range(t + 1, 9):
                    altered = K.copy()
                    altered[future - 1, 0] = sp.log(128)
                    changed_ids, changed_support, _ = block_route(t, q, altered)
                    _, changed_output = restricted_attention(t, q, altered, V, changed_support)
                    self.assertEqual((ids, support), (changed_ids, changed_support))
                    self.assertTrue(zero(out - changed_output))
        self.assertEqual(len(full), 8)  # 28 individual suffix perturbations above.
        _, support, scores = block_route(8, q, K)
        self.assertEqual(tuple(sp.exp(scores[u]) for u in (1, 2, 3)), (3, 4, 1))
        self.assertEqual(support, (3, 4, 7, 8))
        self.assertEqual(block_route(1, q, K, k_b=10)[0], ())
        self.assertEqual(block_route(3, q, K, k_b=10)[0], (1,))
        self.assertEqual(block_route(7, q, K, k_b=0)[1], (7,))
        for b, budget in ((0, 1), (2, -1), (True, 1), (2, 1.0)):
            with self.assertRaises((TypeError, ValueError)):
                block_route(7, q, K, b, budget)

    def test_deliberate_future_leak_and_packed_request_boundaries(self):
        altered = K.copy()
        altered[7, 0] = sp.log(128)
        low = block_route(7, q, K, faulty_current=True)
        high = block_route(7, q, altered, faulty_current=True)
        self.assertEqual((sp.exp(low[2][4]), sp.exp(high[2][4])), (2, 16))
        self.assertEqual(low[:2], ((2,), (3, 4, 7)))
        self.assertEqual(high[:2], ((4,), (7,)))
        self.assertEqual(restricted_attention(7, q, K, V, low[1])[1], sp.Matrix([R(1, 5), 1]))
        self.assertEqual(restricted_attention(7, q, altered, V, high[1])[1], sp.Matrix([1, 1]))
        _, shortK, shortV = fixture((9, 1, 4))
        paddedK = shortK.col_join(sp.Matrix([[sp.log(9999), 0]]))
        paddedV = shortV.col_join(sp.Matrix([[9999, 9999]]))
        requests = {"first": (K, V, 8), "second": (paddedK, paddedV, 3)}
        result = packed_route(requests, "second", 3, q)
        self.assertEqual(result[1], (1, 2, 3))
        self.assertEqual(result[-1], restricted_attention(3, q, shortK, shortV, (1, 2, 3))[1])
        requests["first"] = (altered, 100 * V, 8)
        self.assertEqual(packed_route(requests, "second", 3, q), result)
        self.assertEqual(packed_route(requests, "second", 1, q)[1], (1,))
        with self.assertRaises(ValueError):
            packed_route(requests, "second", 4, q)

    def test_oracles_and_output_error_counterexamples(self):
        blocks = [(diagnostics(block + (7, 8))[0], u) for u, block in enumerate(BLOCKS[:3], 1)]
        self.assertEqual(max(blocks), (R(7, 12), 1))
        pairs = [(diagnostics(ids + (7, 8))[0], ids) for ids in combinations(range(1, 7), 2)]
        self.assertEqual(len(pairs), 15)
        best_mass = max(mass for mass, _ in pairs)
        winner = min(ids for mass, ids in pairs if mass == best_mass)
        self.assertEqual((best_mass, winner), (R(17, 24), (1, 3)))
        equalV = sp.ones(8, 2)
        self.assertEqual(restricted_attention(8, q, K, equalV, (7, 8))[1], sp.Matrix([1, 1]))
        self.assertEqual(diagnostics((7, 8))[1], R(5, 6))
        delta, M = sp.symbols("delta M", positive=True)
        self.assertTrue(zero(((1 - delta) * M + delta * (-M)) - M + 2 * M * delta))
        a = (R(2, 5), R(3, 10), R(3, 10))
        vals = (0, 1, -1)
        self.assertEqual(sum(x * v for x, v in zip(a, vals)), 0)
        self.assertEqual((a[1] * vals[1] + a[2] * vals[2]) / (a[1] + a[2]), 0)
        self.assertEqual((a[0] * vals[0] + a[1] * vals[1]) / (a[0] + a[1]), R(3, 7))

    def test_pooling_exact_factorization_and_multiplicity(self):
        exponential_weights, output = pooled_attention(q, K, V, BLOCKS[:3], (7, 8))
        self.assertEqual(exponential_weights, (3, 4, 1, 2, 2))
        self.assertEqual(output, sp.Matrix([R(11, 24), R(2, 3)]))
        weights2, output2 = pooled_attention(q, K, V, BLOCKS[:3], (7, 8), correct_size=True)
        self.assertEqual(weights2, (6, 8, 2, 2, 2))
        self.assertEqual(output2, sp.Matrix([R(7, 20), R(3, 5)]))
        masses, means = block_statistics(8, q, K, V, BLOCKS)
        self.assertEqual(masses, (10, 8, 2, 4))
        self.assertEqual(means[0], sp.Matrix([R(9, 10), 0]))
        exact = sum((mass * mean for mass, mean in zip(masses, means)), sp.zeros(2, 1)) / sum(masses)
        dense = restricted_attention(8, q, K, V, range(1, 9))[1]
        self.assertTrue(zero(exact - dense))
        self.assertFalse(zero(output - dense))
        self.assertFalse(zero(output2 - dense))
        # Unequal block sizes plus a singleton, equal logits within each block.
        eqK = sp.Matrix([[sp.log(2), 0], [sp.log(2), 0], [sp.log(3), 0]])
        eqV = V[:3, :]
        exact_eq = restricted_attention(3, q, eqK, eqV, (1, 2, 3))[1]
        corrected = pooled_attention(q, eqK, eqV, ((1, 2),), (3,), correct_size=True)[1]
        self.assertTrue(zero(exact_eq - corrected))
        uncorrected = pooled_attention(q, eqK, eqV, ((1, 2),), (3,))[1]
        self.assertFalse(zero(exact_eq - uncorrected))
        eqK4 = sp.Matrix([[sp.log(2), 0], [sp.log(2), 0], [sp.log(3), 0], [sp.log(3), 0]])
        self.assertTrue(zero(pooled_attention(q, eqK4, V[:4, :], ((1, 2), (3, 4)), ())[1]
                             - pooled_attention(q, eqK4, V[:4, :], ((1, 2), (3, 4)), (), correct_size=True)[1]))
        with self.assertRaises(ValueError):
            block_statistics(8, q, K, V, ((1, 2), (2, 3)))
        invalidV = V.copy()
        invalidV[0, 0] = sp.oo
        for values in (V[:7, :], invalidV):
            with self.assertRaises(ValueError):
                block_statistics(8, q, K, values, BLOCKS)
            with self.assertRaises(ValueError):
                pooled_attention(q, K, values, BLOCKS[:3], (7, 8))
        with self.assertRaises(ValueError):
            pooled_attention(q.T, K, V, BLOCKS[:3], (7, 8))
        with self.assertRaises(ValueError):
            pooled_attention(q, K, V, ((1, 2),), (2, 7))
        with self.assertRaises(ValueError):
            pooled_attention(q, K, V, ((),), (7, 8))
        # Slicing the supplied prefix prevents a future neighbor entering a mean.
        changed = K.copy()
        changed[7, 0] = sp.log(128)
        prefix_pool = pooled_attention(q, K[:7, :], V[:7, :], BLOCKS[:3], (7,))
        self.assertEqual(prefix_pool, pooled_attention(q, changed[:7, :], V[:7, :], BLOCKS[:3], (7,)))

    def test_hierarchy_ceiling_and_reused_index_mass(self):
        pool = top_k({1: 0, 2: 2, 3: 1}, 2)
        self.assertEqual(pool, (2, 3))
        best = top_k({u: (10, 8, 2)[u - 1] for u in pool}, 1)
        self.assertEqual(best, (2,))
        self.assertTrue(set(best).issubset(pool))
        self.assertNotIn(1, pool)
        self.assertEqual(diagnostics((3, 4, 7, 8))[0], R(1, 2))
        ceiling = diagnostics((3, 4, 5, 6, 7, 8))[0]
        self.assertEqual(ceiling, R(7, 12))
        for size in range(1, 7):
            for subset in combinations((3, 4, 5, 6, 7, 8), size):
                self.assertLessEqual(diagnostics(subset)[0], ceiling)
        for exponential_weights, expected in (((9, 1), R(9, 10)), ((1, 9), R(1, 10))):
            _, twoK, twoV = fixture(exponential_weights)
            a, _ = restricted_attention(2, q, twoK, twoV, (1, 2))
            self.assertEqual(a[0], expected)  # same reused support {1}, changed layer mass

    def test_top_k_stability_gradient_and_cost_calculus(self):
        self.assertEqual(top_k({3: 1, 2: 4, 1: 4}, 1), (1,))
        self.assertEqual(top_k({1: 1}, 0), ())
        self.assertEqual(top_k({1: sp.Symbol("x", real=True), 2: 0}, 0), ())
        self.assertEqual(top_k({1: sp.Symbol("x", real=True), 2: 0}, 2), (1, 2))
        self.assertEqual(top_k({1: 1}, 9), (1,))
        self.assertEqual(top_k({}, 2), ())
        u = {1: sp.log(10), 2: sp.log(8), 3: sp.log(2)}
        epsilon = sp.log(R(5, 4)) / 4  # strictly below half the exact boundary gap
        for signs in product((-1, 1), repeat=3):
            perturbed = {j: u[j] + signs[j - 1] * epsilon for j in u}
            self.assertEqual(top_k(perturbed, 1), (1,))
        self.assertEqual(top_k({1: 0, 2: 2}, 1), (2,))
        self.assertEqual(top_k({1: 1, 2: 1}, 1), (1,))
        with self.assertRaises(ValueError):
            top_k({1: sp.Symbol("x", real=True), 2: 0}, 1)
        theta = sp.symbols("theta", real=True)
        rho = sp.exp(theta) / (1 + sp.exp(theta))
        loss = -R(3, 4) * sp.log(rho) - R(1, 4) * sp.log(1 - rho)
        self.assertTrue(zero(sp.diff(loss, theta) - rho + R(3, 4)))
        self.assertEqual(sp.diff(loss, theta).subs(theta, 0), -R(1, 4))
        r = sp.Matrix(sp.symbols("r1:4", real=True))
        teacher = (R(1, 2), R(1, 3), R(1, 6))
        z = sum(sp.exp(x) for x in r)
        smooth_loss = -sum(teacher[i] * (r[i] - sp.log(z)) for i in range(3))
        self.assertTrue(all(zero(sp.diff(smooth_loss, r[i]) - sp.exp(r[i]) / z + teacher[i])
                            for i in range(3)))
        alpha, beta, T, b, kp = sp.symbols("alpha beta T b kp", positive=True)
        cost = alpha * T**2 / b + beta * T * (kp + 1) * b
        optimum = sp.sqrt(alpha * T / (beta * (kp + 1)))
        self.assertTrue(zero(sp.diff(cost, b).subs(b, optimum)))
        self.assertTrue(zero(sp.diff(cost, b, 2) - 2 * alpha * T**2 / b**3))
        self.assertTrue(sp.diff(cost, b, 2).is_positive)
        self.assertEqual(optimum.subs({alpha: 1, beta: 1, T: 1024, kp: 3}), 16)
        self.assertEqual([cost.subs({alpha: 1, beta: 1, T: 1024, kp: 3, b: size})
                          for size in (8, 16, 32)], [163840, 131072, 163840])

    def test_exact_counts_exhaustive_declared_small_domain(self):
        self.assertEqual(pair_counts(8, 2, 1), (36, 15, 24, 12))
        cases = 0
        # All 720 combinations: n_b=1..6, b=1..4, k_b=0..5, w=1..5.
        for nb, b, budget, w in product(range(1, 7), range(1, 5), range(6), range(1, 6)):
            T = nb * b
            main_edges, route_edges, local_edges, dense_edges = set(), set(), set(), set()
            for t in range(1, T + 1):
                current = (t - 1) // b + 1
                support = set(range((current - 1) * b + 1, t + 1))
                # Pick first eligible blocks; their equal sizes make identity immaterial.
                for u in list(range(1, current))[:budget]:
                    support.update(range((u - 1) * b + 1, u * b + 1))
                main_edges.update((t, j) for j in support)
                route_edges.update((t, u) for u in range(1, current))
                local_edges.update((t, j) for j in range(max(1, t - w + 1), t + 1))
                dense_edges.update((t, j) for j in range(1, t + 1))
            self.assertEqual(pair_counts(T, b, budget, w),
                             tuple(map(len, (dense_edges, local_edges, main_edges, route_edges))))
            cases += 1
        self.assertEqual(cases, 720)
        with self.assertRaises(ValueError):
            pair_counts(7, 2, 1)


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}; SymPy {sp.__version__}", flush=True)
    unittest.main(verbosity=2)
