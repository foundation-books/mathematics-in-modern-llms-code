"""Independent finite references, invalid interfaces and deliberate mutants."""
import unittest
from itertools import combinations, product
from collections import Counter
import sympy as sp
import extensions_reference as ex

R = sp.Rational


def compositions(total, size, minimum=0):
    if size == 1:
        if total >= minimum:
            yield (total,)
    else:
        for first in range(minimum, total - minimum * (size - 1) + 1):
            for tail in compositions(total - first, size - 1, minimum):
                yield (first,) + tail


def pack_oracle(items, capacities):
    """All labeled assignments through independent choose/remove recursion."""
    if len(capacities) == 1:
        if len(items) == capacities[0]:
            yield (tuple(items),)
        return
    for ids in combinations(range(len(items)), capacities[0]):
        chosen = set(ids)
        head = tuple(items[i] for i in ids)
        rest = tuple(items[i] for i in range(len(items)) if i not in chosen)
        for tail in pack_oracle(rest, capacities[1:]):
            yield (head,) + tail


class GroupSelectionTests(unittest.TestCase):
    def setUp(self):
        self.s = dict(enumerate([10, 1, 9, 8], 1))
        self.g = {1: (1, 2), 2: (3, 4)}

    def test_independent_subsets_and_direct_experts(self):
        sets = list(combinations(self.s, 2))
        sums = [sum(self.s[e] for e in t) for t in sets]
        self.assertEqual(sums, [11, 19, 18, 10, 9, 17])
        feasible = [t for t in sets if t in [(1, 2), (3, 4)]]
        oracle = max(feasible, key=lambda t: sum(self.s[e] for e in t))
        self.assertEqual(ex.constrained_select(self.s, self.g, 2, 1), oracle)
        h = ex.group_select(self.s, self.g, 2, 1)
        self.assertEqual(h, (1, 2))
        self.assertNotEqual(h, oracle)  # kills the heuristic-is-optimal mutant
        z = sp.Matrix([R(2, 3), R(5, 6)])
        outputs = {1: sp.Matrix([z[0], 0]), 2: sp.Matrix([0, z[1]]),
                   3: z, 4: sp.Matrix([2 * z[0], z[1]])}
        for ids, expected in [((1, 3), [R(2, 3), R(15, 38)]),
                              ((1, 2), [R(20, 33), R(5, 66)]),
                              ((3, 4), [R(50, 51), R(5, 6)])]:
            self.assertEqual(ex.selected_mixture(ids, self.s, outputs), sp.Matrix(expected))
        wrong = sum((self.s[e] * outputs[e] for e in h), sp.zeros(2, 1)) / sum(self.s.values())
        self.assertNotEqual(wrong, ex.selected_mixture(h, self.s, outputs))

    def test_bias_never_enters_coefficients(self):
        affinities = {1: 10, 2: 1}
        outputs = {1: sp.Matrix([1, 0]), 2: sp.Matrix([0, 1])}
        self.assertEqual(ex.selected_mixture((1, 2), affinities, outputs), sp.Matrix([R(10, 11), R(1, 11)]))
        self.assertNotEqual(ex.selected_mixture((1, 2), {1: 20, 2: 1}, outputs),
                            ex.selected_mixture((1, 2), affinities, outputs))

    def test_partition_and_feasibility_errors(self):
        for groups in [{1: (1, 2), 2: (2, 3, 4)}, {1: (1, 2)},
                       {1: (), 2: (1, 2, 3, 4)}]:
            with self.assertRaises(ValueError):
                ex.group_select(self.s, groups, 2, 1)
        with self.assertRaises(ValueError):
            ex.group_select(self.s, {1: (1,), 2: (2, 3, 4)}, 2, 1)
        self.assertEqual(ex.constrained_select(self.s, {1: (1,), 2: (2, 3, 4)}, 2, 1), (3, 4))
        with self.assertRaises(ValueError):
            ex.constrained_select(self.s, self.g, 3, 1)
        for k, g in [(0, 1), (5, 1), (2, 0), (2, 3), (True, 1)]:
            with self.assertRaises(ValueError):
                ex.group_select(self.s, self.g, k, g)

    def test_ties_identity_and_all_groups(self):
        tie = {1: 2, 2: 2, 3: 2, 4: 1}
        self.assertEqual(ex.group_select(tie, {2: (4, 3), 1: (2, 1)}, 2, 1), (1, 2))
        self.assertEqual(ex.group_select(self.s, self.g, 2, 2), (1, 3))
        self.assertEqual(ex.topk(tie, [4, 3, 2, 1], 2), (1, 2))

    def test_finite_domain_monotonicity_against_enumeration(self):
        for values in product(range(3), repeat=4):
            scores = dict(enumerate(values, 1))
            for k in [1, 2]:
                all_sets = list(combinations(range(1, 5), k))
                constrained = [t for t in all_sets if all(e <= 2 for e in t) or all(e > 2 for e in t)]
                expected = max(sum(values[e - 1] for e in t) for t in constrained)
                got = ex.constrained_select(scores, self.g, k, 1)
                self.assertEqual(sum(scores[e] for e in got), expected)
                larger = ex.constrained_select(scores, self.g, k, 2)
                self.assertLessEqual(expected, sum(scores[e] for e in larger))


class ReplicaTests(unittest.TestCase):
    def test_threshold_independent_count_oracle(self):
        weights = [70, 60, 50, 40]
        allocations = list(compositions(8, 4, 1))
        self.assertEqual(len(allocations), 35)
        maxima = [max(R(w, r) for w, r in zip(weights, counts)) for counts in allocations]
        self.assertEqual(min(maxima), 35)
        self.assertEqual(ex.replica_counts(weights, 8), (2, 2, 2, 2))
        for tau in [0, R(349, 10), 35, 40, 55, 70]:
            self.assertEqual(ex.threshold_feasible(weights, 8, tau), any(v <= tau for v in maxima))
        for weights in product(range(4), repeat=3):
            for slots in range(3, 7):
                rs = list(compositions(slots, 3, 1))
                for tau in [0, R(1, 2), 1, 2, 3]:
                    brute = any(all(R(w, r) <= tau for w, r in zip(weights, counts)) for counts in rs)
                    self.assertEqual(ex.threshold_feasible(weights, slots, tau), brute)

    def test_all_packings_global_and_node_oracle(self):
        items = (35, 35, 30, 30, 25, 25, 20, 20)
        oracle = min(max(map(sum, packs)) for packs in pack_oracle(items, (2, 2, 2, 2)))
        self.assertEqual(oracle, 55)
        for devices, expected in [([(1, 1), (2, 2), (3, 3), (4, 4)], (70, 60, 50, 40)),
                                  ([(1, 4), (1, 4), (2, 3), (2, 3)], (55,) * 4),
                                  ([(1, 2), (1, 2), (3, 4), (3, 4)], (65, 65, 45, 45))]:
            result = ex.packing([70, 60, 50, 40], [2] * 4, devices, [2] * 4)
            self.assertEqual(result['loads'], expected)
            self.assertEqual(sum(result['loads']), 220)
            for e, physical in result['logical_to_physical'].items():
                self.assertEqual(len(physical), 2)
                self.assertTrue(all(result['physical_to_logical'][p] == e for p in physical))
        # Independently enumerate each node's four-copy pairings.
        node_a = min(max(map(sum, p)) for p in pack_oracle((35, 35, 30, 30), (2, 2)))
        node_b = min(max(map(sum, p)) for p in pack_oracle((25, 25, 20, 20), (2, 2)))
        self.assertEqual((node_a, node_b), (65, 45))

    def test_locality_capacity_memory_and_version_errors(self):
        args = ([70, 60, 50, 40], [2] * 4, [(1, 4), (1, 4), (2, 3), (2, 3)], [2] * 4)
        with self.assertRaises(ValueError):
            ex.packing(*args, device_nodes=[0, 0, 1, 1], allowed_nodes={1: {0}, 2: {0}, 3: {1}, 4: {1}})
        with self.assertRaises(ValueError):
            ex.packing(*args, versions=['v0'] * 7 + ['stale'])
        with self.assertRaises(ValueError):
            ex.packing(*args, sizes=[4] * 4, memory=[7] * 4)
        with self.assertRaises(ValueError):
            ex.packing(*args[:3], [1, 2, 2, 3])
        with self.assertRaises(ValueError):
            ex.packing([1, 1], [1, 1], [(1, 1)], [2])
        valid = ex.packing([2, 4], [1, 2], [(1,), (2, 2)], [1, 2], sizes=[4, 8], memory=[4, 16])
        self.assertEqual(valid['loads'], (2, 4))
        self.assertEqual(valid['parameter_bytes'], (4, 16))

    def test_zero_and_invalid_budget(self):
        self.assertEqual(ex.replica_counts([0, 0], 3), (2, 1))
        self.assertTrue(ex.threshold_feasible([0, 0], 3, 0))
        self.assertFalse(ex.threshold_feasible([0, 0], 1, 0))
        for weights, slots in [([1, 2], 1), ([-1, 2], 3), ([], 3)]:
            with self.assertRaises(ValueError):
                ex.replica_counts(weights, slots)

    def test_greedy_counterexample_and_nonuniform_dispatch(self):
        w = [8, 7, 6, 5, 4, 2]
        packs = ex.greedy_packs(w, [3, 3])
        self.assertEqual(tuple(sum(w[i] for i in p) for p in packs), (17, 15))
        optimum = min(max(sum(w[i] for i in ids), sum(w) - sum(w[i] for i in ids))
                      for ids in combinations(range(6), 3))
        self.assertEqual(optimum, 16)
        ideal = ex.packing([70, 60, 50, 40], [2] * 4, [(1, 4), (1, 4), (2, 3), (2, 3)], [2] * 4)['loads']
        actual = (70 + 20, 0 + 20, 30 + 25, 30 + 25)
        self.assertNotEqual(actual, ideal)
        self.assertEqual(sum(actual), sum(ideal))


if __name__ == '__main__':
    unittest.main()
