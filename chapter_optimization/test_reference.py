"""Behavioral checks against Chapter 14's mathematical specification."""
import unittest
from unittest.mock import patch

import sympy as sp

from reference import (A, A0, DA, G1, G2, GBAR, M, MH, PRECISION, QBAR,
                       READOUT, O_STAR, Parameter, adam_direction, equal, frob,
                       direction_comparison_rows, power_transform, q, run_checks, supported_polar,
                       symbolic_gradients, update_tied_uses)


class OptimizerContractTests(unittest.TestCase):
    def test_exact_anchors_and_deliberate_errors(self):
        checks = run_checks()
        self.assertGreaterEqual(len(checks), 12)
        self.assertEqual(len(checks), len(set(checks)))

    def test_chain_rule_away_from_zero(self):
        _, gradients, _, _ = symbolic_gradients()
        entries = sp.symbols('w11 w12 w21 w22', real=True)
        weights = sp.Matrix([[q(1, 5), -q(1, 7)], [q(2, 9), q(1, 4)]])
        logits = READOUT*weights*O_STAR
        probabilities = logits.applyfunc(sp.exp)/sum(sp.exp(x) for x in logits)
        expected = READOUT.T*(probabilities-sp.Matrix([0, 1, 0, 0]))*O_STAR.T
        actual = gradients[0].subs(dict(zip(entries, weights)))
        self.assertTrue(equal(actual, expected))

    def test_exact_orthogonal_basis_equivariance(self):
        left, right = sp.Matrix([[0, -1], [1, 0]]), sp.diag(1, -1)
        for mat in (GBAR, G1, M, MH):
            self.assertTrue(equal(supported_polar(left*mat*right.T),
                                  left*supported_polar(mat)*right.T))

    def test_norm_ball_attainment(self):
        for mat, nuclear in ((M, sp.Integer(6)), (G1, frob(G1))):
            polar = supported_polar(mat)
            self.assertTrue(equal(sp.trace(mat.T*(-polar)), -nuclear))
            # Nonzero singular directions of a supported factor have norm one.
            gram = polar.T*polar
            self.assertTrue(equal(gram*gram, gram))
            normalized = -mat/frob(mat)
            self.assertTrue(equal(frob(normalized), 1))
            self.assertTrue(equal(sp.trace(mat.T*normalized), -frob(mat)))

    def test_absence_preserves_existing_state_but_zero_evolves_it(self):
        parameter = Parameter('internal', sp.ones(2), 'matrix')
        parameter.step(GBAR, exact=True)
        old_value, old_state = parameter.value.copy(), parameter.momentum.copy()
        parameter.step(None, exact=True)
        self.assertEqual(parameter.count, 1)
        self.assertTrue(equal(parameter.value, old_value))
        self.assertTrue(equal(parameter.momentum, old_state))
        parameter.step(sp.zeros(2), exact=True)
        self.assertEqual(parameter.count, 2)
        self.assertTrue(equal(parameter.momentum, old_state/2))
        expected = q(49, 50)*old_value-q(1, 10)*sp.sqrt(2)*QBAR
        self.assertTrue(equal(parameter.value, expected))

    def test_conflicting_tied_storage_rejected_before_update(self):
        owner = Parameter('embedding', sp.zeros(2), 'adamw')
        conflict = Parameter('readout', sp.zeros(2), 'matrix')
        conflict.value = owner.value
        with self.assertRaisesRegex(ValueError, 'contradictory'):
            update_tied_uses([(owner, G1), (conflict, G2)])
        self.assertEqual(owner.count, 0)
        self.assertEqual(conflict.count, 0)

    def test_invalid_shapes_and_iteration_contract(self):
        for steps in (-1, q(1, 2)):
            with self.assertRaises(ValueError):
                power_transform(A, steps)
        with self.assertRaises(ValueError):
            power_transform(A0, polynomial='undefined')
        parameter = Parameter('projection', sp.zeros(2), 'matrix')
        with self.assertRaises(ValueError):
            parameter.step(sp.zeros(2, 3))
        self.assertEqual(parameter.count, 0)

    def test_adam_after_multiple_absent_steps_uses_own_counter(self):
        parameter = Parameter('tied', sp.zeros(2), 'adamw')
        for _ in range(3):
            parameter.step(None)
        parameter.step(GBAR)
        self.assertTrue(equal(parameter.value, -DA/10))
        parameter.step(None)
        direction = adam_direction(G1, parameter.first, parameter.second, 2)[0]
        expected = q(49, 50)*parameter.value-direction/10
        parameter.step(G1)
        self.assertEqual(parameter.count, 2)
        self.assertTrue(equal(parameter.value, expected))

    def test_polar_magnitude_detects_a_wrong_sign(self):
        # A negative factor still reconstructs A through a symmetric magnitude
        # and has the same support Grams. Positivity must reject that certificate.
        def wrong_factor(mat):
            polar = supported_polar(mat)
            return -polar if equal(mat, A) else polar

        with patch('reference.supported_polar', side_effect=wrong_factor):
            with self.assertRaisesRegex(AssertionError, 'wide positive semidefinite'):
                run_checks()

    def test_task_f_compares_both_polynomials_and_complete_changes(self):
        rows = direction_comparison_rows()
        self.assertEqual([row['method'] for row in rows],
                         ['SGD', 'Adam', 'exact polar', 'cubic', 'quintic'])
        by_method = {row['method']: row for row in rows}
        self.assertTrue(equal(by_method['SGD']['raw_direction'], GBAR))
        self.assertTrue(equal(by_method['Adam']['raw_direction'], DA))
        self.assertTrue(equal(by_method['exact polar']['raw_direction'], QBAR))
        for row in rows:
            self.assertTrue(equal(row['direction'], row['scale']*row['raw_direction']))
            self.assertTrue(equal(row['loss_change'], -row['direction']/10))
            self.assertTrue(equal(row['decay_change'], sp.zeros(2)))
            self.assertTrue(equal(row['parameter'], row['loss_change']+row['decay_change']))
        for method in ('cubic', 'quintic'):
            self.assertEqual(by_method[method]['iterations'], 5)
            self.assertTrue(equal(by_method[method]['scale'], sp.sqrt(2)))
        # Independently evaluate the declared quintic recurrence on this fixture.
        expected = (GBAR/frob(GBAR)).evalf(PRECISION)
        for _ in range(5):
            gram = expected*expected.T
            expected = (q(6889, 2000)*expected-q(191, 40)*gram*expected
                        + q(4063, 2000)*gram*gram*expected).evalf(PRECISION)
        error = frob(by_method['quintic']['raw_direction']-expected).evalf(PRECISION)
        self.assertLess(error, q(1, 10**30))


if __name__ == '__main__':
    unittest.main()
