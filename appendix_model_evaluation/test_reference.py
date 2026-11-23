"""Behavioral and independently enumerated checks for Appendix A."""

from math import comb
from pathlib import Path
import re
import unittest

import sympy as sp

import reference as ref


class EvaluationChecks(unittest.TestCase):
    def test_reversing_systems_reverses_effect_but_preserves_variance_and_test(self):
        forward = ref.paired_statistics((8, 6, 2, 4))
        reverse = ref.paired_statistics((8, 2, 6, 4))
        self.assertEqual(reverse["mean"], -forward["mean"])
        for key in ("variance", "se2", "covariance"):
            self.assertEqual(reverse[key], forward[key])
        self.assertEqual(ref.exact_pvalue(6, 2), ref.exact_pvalue(2, 6))

    def test_exact_test_conditional_size_and_symmetry_through_twelve_discordances(self):
        for m in range(13):
            for a in range(m + 1):
                b = m - a
                # Enumerate both inclusive tails independently of the smaller-count formula.
                lower = sp.Rational(sum(comb(m, j) for j in range(a + 1)), 2**m)
                upper = sp.Rational(sum(comb(m, j) for j in range(a, m + 1)), 2**m)
                self.assertEqual(ref.exact_pvalue(a, b), min(1, 2*min(lower, upper)))
                self.assertEqual(ref.exact_pvalue(a, b), ref.exact_pvalue(b, a))
            for alpha in (sp.Rational(1, 20), sp.Rational(1, 200)):
                size = sum(sp.Rational(comb(m, a), 2**m) for a in range(m + 1)
                           if ref.exact_pvalue(a, m - a) <= alpha)
                self.assertLessEqual(size, alpha)
        self.assertEqual(ref.exact_pvalue(0, 0), 1)
        self.assertEqual(ref.exact_pvalue(4, 4), 1)

    def test_document_aggregation_preserves_counts_and_changes_se(self):
        self.assertEqual(tuple(map(sum, zip(*ref.DOCUMENT_COUNTS))), ref.PAIRED_COUNTS)
        means, (mean, variance, se2) = ref.document_statistics()
        self.assertEqual(means, [0, sp.Rational(2, 5), sp.Rational(3, 5), -sp.Rational(1, 5)])
        self.assertEqual((mean, variance, se2), (sp.Rational(1, 5), sp.Rational(2, 15), sp.Rational(1, 30)))
        self.assertNotEqual(se2, ref.paired_statistics()["se2"])

    def test_same_marginals_can_have_opposite_covariance_signs(self):
        for pairs, covariance, variance in [([(0, 0), (1, 1)], sp.Rational(1, 4), 0),
                                             ([(1, 0), (0, 1)], -sp.Rational(1, 4), 1)]:
            self.assertEqual(sp.Rational(sum(a for a, _ in pairs), 2), sp.Rational(1, 2))
            cov = sum((sp.Rational(a) - sp.Rational(1, 2)) *
                      (sp.Rational(b) - sp.Rational(1, 2)) for a, b in pairs)/2
            self.assertEqual(cov, covariance)
            self.assertEqual(sp.Rational(sum((a-b)**2 for a, b in pairs), 2), variance)

    def test_strict_dominance_and_utility_regions_including_every_crossing(self):
        self.assertFalse(ref.dominates((1, 2), (1, 2)))
        self.assertTrue(ref.dominates((1, 2), (sp.Rational(9, 10), 2)))
        self.assertEqual(ref.pareto_set(), {"A", "B", "D", "E"})
        regions = ref.utility_regions()
        weights = {sp.Rational(0)}
        for q, c in ref.POINTS.values():
            for q2, c2 in ref.POINTS.values():
                if c != c2 and (q-q2)/(c-c2) >= 0:
                    weights.add((q-q2)/(c-c2))
        ordered = sorted(weights)
        witnesses = ordered + [(a+b)/2 for a, b in zip(ordered, ordered[1:])] + [ordered[-1] + 1]
        for weight in witnesses:
            actual = ref.utility_winners(weight)
            self.assertEqual(actual, {name for name, region in regions.items() if weight in region})
            self.assertNotIn("E", actual)
        self.assertEqual(ref.utility_winners(sp.Rational(1, 80)), {"D", "A"})
        self.assertEqual(ref.utility_winners(sp.Rational(1, 40)), {"A", "B"})
        self.assertEqual(ref.utility_winners(sp.Rational(1, 60)), {"A"})
        self.assertEqual(ref.cap_winners(3), {"E"})
        self.assertEqual(ref.cap_winners(1), set())

    def test_invalid_counts_and_small_samples_are_rejected(self):
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                ref.exact_pvalue(invalid, 0)
        with self.assertRaises(ValueError):
            ref.pairs_from_counts((1, 2, 3))
        with self.assertRaises(ValueError):
            ref.paired_statistics((0, 0, 0, 0))
        with self.assertRaises(ValueError):
            ref.sample_statistics([1])

    def test_printed_appendix_inputs_match_companion(self):
        source = Path(__file__).resolve().parents[2]/"book/appendix_model_evaluation.tex"
        if not source.exists():
            self.skipTest("Standalone companion copy has no manuscript")
        text = source.read_text()
        for value in ("n_{11}=8", "n_{10}=6", "n_{01}=2", "n_{00}=4",
                      r"\mu_{00}=3/5", r"\mu_{01}=13/20", r"\mu_{10}=7/10", r"\mu_{11}=4/5"):
            self.assertIn(value, text)
        tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
        self.assertCountEqual(
            re.findall(r"\\label\{(tab:eval-[^}]+)\}", "\n".join(tables)),
            ["tab:eval-paired", "tab:eval-factorial", "tab:eval-pareto"],
        )
        self.assertCountEqual(
            re.findall(r"\\caption\*\{([^}]+)\}", "\n".join(tables)),
            ["Evaluation tools and their main-text applications.",
             "Expected coverage for the two hypothetical checkpoints."],
        )
        for figure in ("coverage-versus-returned-accuracy", "evaluation-sampling-units"):
            self.assertIn(r"\label{fig:" + figure + "}", text)


if __name__ == "__main__":
    unittest.main()
