"""Behavior checks beyond the built-in acceptance calculations."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import sympy as sp

import ch01_examples as lab


class LaboratoryTests(unittest.TestCase):
    def test_reference_acceptance(self):
        result = lab.run_checks()
        self.assertEqual(result["invalid_cases"], 21)
        self.assertEqual(result["round_trips"], 40)

    def test_unselected_impossible_target_is_not_evaluated(self):
        rows = dict(lab.ROWS)
        rows[()] = (sp.S.Zero, sp.S.One, sp.S.Zero, sp.S.Zero)
        with patch.object(lab, "ROWS", rows):
            completion = lab.score(mask=lab.COMPLETION_TARGETS)
            self.assertEqual(completion.probability_product, sp.Rational(1, 8))
            self.assertTrue(lab.equivalent(completion.total_nats, 3 * sp.log(2)))
            self.assertEqual(lab.score().total_nats, sp.oo)

    def test_default_context_and_eos_only_sequence(self):
        result = lab.score(("C", "A", "<eos>"))
        self.assertEqual(result.probability_product, sp.Rational(1, 128))
        eos = ("<eos>",)
        self.assertEqual(lab.encode(eos), "111")
        self.assertEqual(lab.decode("111"), eos)

    def test_partial_code_and_trailing_code_are_rejected(self):
        for bits in ("", "11", "010", "01000", "101", "x"):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                lab.decode(bits)

    def test_fresh_run_does_not_depend_on_prior_scores(self):
        first = lab.score(mask=lab.COMPLETION_TARGETS)
        lab.score()
        self.assertEqual(first, lab.score(mask=lab.COMPLETION_TARGETS))
        self.assertNotEqual(first, replace(first, selected=(0, 1, 2)))

    def test_unknown_support_requires_instantiation(self):
        x = sp.Symbol("x", nonnegative=True)
        q = (x / (1 + x), 1 / (1 + x))
        for measure in (lambda: lab.entropy(q),
                        lambda: lab.cross_entropy(q, (sp.Rational(1, 2),) * 2),
                        lambda: lab.kl(q, (sp.Rational(1, 2),) * 2)):
            with self.assertRaisesRegex(ValueError, "Support is unknown"):
                measure()
        self.assertEqual(lab.entropy(tuple(v.subs(x, 0) for v in q)), 0)


if __name__ == "__main__":
    unittest.main()
