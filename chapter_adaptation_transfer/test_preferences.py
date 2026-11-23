"""Focused checks for fixture agreement and mathematical domain failures."""
import unittest
import sympy as sp
from preferences import (CURRENT_TREE, PI_REF, REWARDS, complete_response_distribution,
                       kl, sequence_score, tilted_policy, run_checks)


class PreferenceLaboratoryTests(unittest.TestCase):
    def test_complete_symbolic_laboratory(self):
        report = run_checks()
        self.assertEqual(report['B_optimal_policy'], [sp.Rational(12,17),sp.Rational(4,17),sp.Rational(1,17)])
        self.assertNotEqual(report['F_sequence_margins']['full'],report['F_sequence_margins']['mean'])

    def test_tilt_rejects_invalid_domains(self):
        for coefficient in (0, -1, sp.symbols('unconstrained')):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                tilted_policy(PI_REF, REWARDS, coefficient)
        for reference in ([1,1,1], [sp.Rational(3,5),sp.Rational(2,5),0], [-1,1,1]):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                tilted_policy(reference, REWARDS, 1)
        with self.assertRaises(ValueError):
            tilted_policy(PI_REF, [0,0], 1)
        with self.assertRaises(ValueError):
            tilted_policy(PI_REF, [0,0,sp.oo], 1)

    def test_kl_zero_mass_and_direction(self):
        self.assertEqual(kl([1,0], [1,0]),0)
        self.assertEqual(kl([0,1], [1,0]),sp.oo)
        p, q = [sp.Rational(3,4),sp.Rational(1,4)], [sp.Rational(1,2)]*2
        self.assertNotEqual(kl(p,q),kl(q,p))

    def test_sequence_score_requires_aligned_positive_factors(self):
        with self.assertRaises(ValueError):
            sequence_score([sp.Rational(1,2)], [sp.Rational(1,2),sp.Rational(1,2)])
        with self.assertRaises(ValueError):
            sequence_score([0],[1])
        with self.assertRaises(ValueError):
            sequence_score([1],[1],mode='undeclared')

    def test_complete_tree_rejects_truncation_and_missing_mass(self):
        with self.assertRaises(ValueError):
            complete_response_distribution(CURRENT_TREE, horizon=3)
        incomplete = {prefix: dict(row) for prefix, row in CURRENT_TREE.items()}
        del incomplete[('b', 'c', 'd')]
        with self.assertRaises(ValueError):
            complete_response_distribution(incomplete)
        incomplete = {prefix: dict(row) for prefix, row in CURRENT_TREE.items()}
        del incomplete[()]['d']
        with self.assertRaises(ValueError):
            complete_response_distribution(incomplete)


if __name__ == '__main__':
    unittest.main()
