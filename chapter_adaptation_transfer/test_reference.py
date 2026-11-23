"""Behavioral checks for the mathematical reference and assertion failures."""
import unittest

import sympy as sp

import reference as ref


class ReferenceBehavior(unittest.TestCase):
    def test_named_fixture_inventory(self):
        self.assertEqual(len(ref.checks), 73)
        self.assertEqual(len(set(ref.checks)), 73)

    def test_common_logit_shift_preserves_distribution(self):
        shift = sp.symbols("shift", real=True)
        moved = ref.softmax(ref.teacher_logits + sp.ones(3, 1) * shift)
        self.assertEqual([ref.clean(x) for x in moved - ref.p_teacher], [0, 0, 0])

    def test_kl_rejects_different_event_counts(self):
        with self.assertRaises(ValueError):
            ref.kl([sp.S.One], [sp.Rational(1, 2), sp.Rational(1, 2)])

    def test_false_symbolic_identity_fails(self):
        count = len(ref.checks)
        with self.assertRaises(AssertionError):
            ref.eq("deliberately false", ref.J.diff(ref.u), ref.J_frozen.diff(ref.u))
        self.assertEqual(len(ref.checks), count)

    def test_wrong_matrix_shape_fails(self):
        with self.assertRaises(AssertionError):
            ref.eq("wrong orientation", ref.o_star, ref.o_star.T)

    def test_wrong_prefix_weights_change_sequence_value(self):
        wrong_forward = ref.f + ref.f / 2
        right_forward = ref.kl(ref.P, ref.Q).subs(ref.u, sp.Rational(1, 2))
        self.assertNotEqual(ref.clean(wrong_forward - right_forward), 0)
        wrong_reverse = ref.c + ref.c / 4
        right_reverse = ref.kl(ref.Q, ref.P).subs(ref.u, sp.Rational(1, 2))
        self.assertNotEqual(ref.clean(wrong_reverse - right_reverse), 0)

    def test_parameter_averaging_differs_from_probability_mixing(self):
        logits1 = sp.Matrix([0, 0])
        logits2 = sp.Matrix([sp.log(3), 0])
        law_mean = (ref.softmax(logits1) + ref.softmax(logits2)) / 2
        parameter_mean = ref.softmax((logits1 + logits2) / 2)
        self.assertEqual(law_mean, sp.Matrix([sp.Rational(5, 8), sp.Rational(3, 8)]))
        self.assertNotEqual(ref.clean(law_mean[0] - parameter_mean[0]), 0)


if __name__ == "__main__":
    unittest.main()
