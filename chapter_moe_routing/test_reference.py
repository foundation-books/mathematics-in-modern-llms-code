"""Behavioral checks for semantics, ties, identity and scope counterexamples."""
from dataclasses import replace
import unittest
import sympy as sp
from fixture import BIAS_DEMO, BIAS_ZERO, EXPERTS, MARGINS, R, S, SHARED, TIED_MARGINS, X, exact_topk
from reference import (
    balanced_assignments, combine, dual_bound, hinge, layer, quantile_candidate,
    reference_checks, sign_step, threshold_interval, wrong_biased_weights,
)


class RoutingChecks(unittest.TestCase):
    def test_reference_obligations(self):
        reference_checks()

    def test_ties_and_rejected_symbolic_sort(self):
        self.assertEqual(exact_topk([R(1, 2), R(1, 2), R(1, 3)], 2), (0, 1))
        for k in (0, 4, True):
            with self.assertRaises(ValueError):
                exact_topk([1, 2, 3], k)
        for values in ([sp.Symbol("x"), 1], [0.5, 1], [sp.oo, 1]):
            with self.assertRaises(TypeError):
                exact_topk(values, 1)

    def test_weight_mutation_is_not_detected_by_unit_sum(self):
        result = layer(bias=BIAS_DEMO)
        ids = result.selected[0]
        wrong = wrong_biased_weights(S[0, :], BIAS_DEMO, ids)
        self.assertEqual(sum(wrong), 1)
        wrong_row = X[0, :] * SHARED
        for j, coefficient in zip(ids, wrong):
            wrong_row += coefficient * X[0, :] * EXPERTS[j]
        self.assertEqual(wrong_row[0], R(29, 14))
        self.assertNotEqual(wrong_row, result.output[0, :])

    def test_common_offset_and_frozen_row_independence(self):
        result = layer(bias=BIAS_DEMO)
        shifted = layer(bias=BIAS_DEMO + sp.ones(1, 3) * R(11, 7))
        self.assertEqual(result.weights, shifted.weights)
        self.assertEqual(result.output, shifted.output)
        for i in range(3):
            row = layer(X[i, :], S[i, :], BIAS_DEMO)
            self.assertEqual(row.output, result.output[i, :])
        altered = S.copy()
        altered[2, :] = sp.Matrix([[R(1, 10), R(9, 10), R(8, 10)]])
        before = max(balanced_assignments()[1], key=lambda entry: entry[1])[0]
        after = max(balanced_assignments(altered)[1], key=lambda entry: entry[1])[0]
        self.assertNotEqual(before[1], after[1])  # unchanged row two, changed offline optimum

    def test_record_permutation_and_origin_mutation(self):
        result = layer(bias=BIAS_DEMO)
        self.assertEqual(combine(reversed(result.records), X, EXPERTS, SHARED), result.output)
        records = list(result.records)
        # Swap origins between slot-zero records; expert counts and record count survive.
        records[0] = replace(records[0], origin=1)
        records[2] = replace(records[2], origin=0)
        self.assertEqual(sorted(r.expert for r in records), sorted(r.expert for r in result.records))
        self.assertNotEqual(combine(records, X, EXPERTS, SHARED), result.output)
        self.assertNotEqual(set(records), set(result.records))

    def test_feedback_and_exact_quantile_surrogate_limits(self):
        step = sign_step(BIAS_ZERO, layer().loads, R(1, 10))
        self.assertEqual(step, sp.Matrix([[R(-1, 10), 0, R(1, 10)]]))
        self.assertNotEqual(step, BIAS_DEMO)
        cutoff, margins, candidate = quantile_candidate(S, BIAS_ZERO, 2)
        self.assertEqual(cutoff, sp.Matrix([R(1, 10), R(1, 5), R(3, 10)]))
        self.assertEqual(margins, S - cutoff * sp.ones(1, 3))
        self.assertEqual(candidate, sp.Matrix([[R(-4, 15), R(2, 15), R(2, 15)]]))
        self.assertEqual(layer(bias=candidate).loads, (3, 2, 1))
        _, raw_margins, centered = quantile_candidate(S, BIAS_DEMO, 2)
        self.assertEqual(layer(bias=centered).output, layer(bias=BIAS_DEMO).output)
        wrong_margins = raw_margins + sp.ones(3, 1) * BIAS_DEMO
        wrong_tau = [threshold_interval(tuple(wrong_margins[:, j]), 2)[0] for j in range(3)]
        self.assertEqual(wrong_tau, [0, 0, 0])  # adding the old bias twice corrupts the rule
        self.assertNotEqual(layer(bias=BIAS_ZERO).output, layer(bias=centered).output)
        self.assertEqual(quantile_candidate(S, BIAS_DEMO, 3)[0:2], (None, None))

    def test_tied_and_endpoint_thresholds(self):
        for t in (R(1, 10), R(1, 4), R(1, 2)):
            self.assertEqual(hinge(MARGINS, 2, t), R(13, 10))
        self.assertEqual(hinge(TIED_MARGINS, 2, R(1, 2)), R(13, 10))
        self.assertEqual(sum(1 for m in TIED_MARGINS if m > R(1, 2)), 1)
        self.assertEqual(sum(1 for m in TIED_MARGINS if m >= R(1, 2)), 3)
        self.assertEqual(threshold_interval(MARGINS, 0), (R(4, 5), sp.oo))
        self.assertEqual(threshold_interval(MARGINS, 4), (-sp.oo, R(-2, 5)))
        with self.assertRaises(ValueError):
            balanced_assignments(sp.ones(3, 4), 2)
        tied_scores = sp.ones(3, 3)
        self.assertEqual(layer(scores=tied_scores).loads, (3, 3, 0))
        self.assertEqual(len(balanced_assignments(tied_scores)[1]), 6)

    def test_normalization_placement(self):
        aggregate = 3 / sp.sqrt(13)
        expertwise = 1 / (2 * sp.sqrt(2)) + 1 / sp.sqrt(5)
        difference = sp.simplify(aggregate - expertwise)
        self.assertNotEqual(difference, 0)
        self.assertGreater(float(difference.evalf()), 0)
        # The mathematical proof uses strict concavity on positive inputs.
        v = sp.Symbol("v", positive=True)
        norm = v / sp.sqrt(v ** 2 + 1)
        self.assertEqual(sp.simplify(sp.diff(norm, v, 2)), -3 * v / (v ** 2 + 1) ** R(5, 2))


if __name__ == "__main__":
    unittest.main()
