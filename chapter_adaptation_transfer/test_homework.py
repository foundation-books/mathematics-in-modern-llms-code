"""Current assignment coverage and distinction between the constructed cases."""
import unittest

import sympy as sp

from homework_reference import run_homework


class IntegratedHomeworkTests(unittest.TestCase):
    def test_all_four_parts_and_protected_results(self):
        report = run_homework()
        self.assertEqual(set(report), {'A', 'B', 'C', 'D', 'checks'})
        self.assertEqual(report['A']['first_step_logits'],
                         [sp.Rational(20, 27), sp.Rational(18, 27), sp.Rational(16, 27)])
        self.assertEqual(report['C']['optimal_policy'],
                         [sp.Rational(12, 17), sp.Rational(4, 17), sp.Rational(1, 17)])
        self.assertNotEqual(report['D']['margins']['full'], report['D']['margins']['mean'])
        self.assertEqual(report['D']['unobserved_mass'], [sp.Rational(2, 3), sp.Rational(1, 3)])


if __name__ == '__main__':
    unittest.main()
