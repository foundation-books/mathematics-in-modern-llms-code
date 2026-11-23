"""Focused behavior checks for support semantics and shared-input gradients."""

import unittest

import sympy as sp

from reference import equal, masked_softmax, matrix_directional_check, stable_float_softmax


class AttentionChecks(unittest.TestCase):
    def test_invalid_domain_is_rejected(self):
        for scores, temperature, support in [
            (sp.Matrix([0, 1]), 1, ()),
            (sp.Matrix([0, 1]), 1, (0, 0)),
            (sp.Matrix([0, 1]), 1, (2,)),
            (sp.Matrix([0, 1]), 0, None),
            (sp.Matrix([0, 1]), -1, None),
            (sp.Matrix([0, sp.oo]), 1, None),
            (sp.Matrix([0, sp.nan]), 1, None),
        ]:
            with self.subTest(scores=scores, temperature=temperature, support=support):
                with self.assertRaises(ValueError):
                    masked_softmax(scores, temperature, support)
        for logits in ([], [float("inf")], [float("nan")]):
            with self.assertRaises(ValueError):
                stable_float_softmax(logits)

    def test_future_score_and_value_cannot_change_prefix(self):
        original = sp.Matrix([0, sp.log(2), sp.log(3)])
        changed = sp.Matrix([0, sp.log(2), 1000])
        values = sp.Matrix([[1, 0], [0, 1], [1, 1]])
        changed_values = sp.Matrix([[1, 0], [0, 1], [999, -999]])
        for allowed in ((0,), (0, 1)):
            with self.subTest(allowed=allowed):
                a = masked_softmax(original, allowed=allowed)
                other = masked_softmax(changed, allowed=allowed)
                equal(a, other)
                equal(values.T * a, changed_values.T * other)

    def test_support_order_does_not_change_distribution(self):
        scores = sp.Matrix([0, sp.log(2), sp.log(3)])
        equal(masked_softmax(scores, 2, (0, 2)), masked_softmax(scores, 2, (2, 0)))

    def test_matrix_gradient_accumulates_all_shared_paths(self):
        matrix_directional_check()


if __name__ == "__main__":
    unittest.main()
