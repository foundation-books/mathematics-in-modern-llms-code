"""Behavioral and mathematical boundaries beyond the main printed fixture."""
import unittest

import sympy as sp

from reference import (Q, tree, weighted_score, group_coefficients,
                       importance_weights, gae, td_target, reductions, verify)


class Chapter19Tests(unittest.TestCase):
    def test_complete_finite_verification(self):
        report = verify()
        self.assertEqual(report['ordered_groups'], 256)
        self.assertEqual(report['numerical_checks'], 1)
        self.assertEqual(report['exact_checks'], len(set(report['named_checks'])))

    def test_gradient_on_other_interior_policies(self):
        p, q, c = sp.symbols('p q c', real=True)
        probabilities, rewards, scores = tree(p, q, c)
        objective = probabilities.dot(rewards)
        derivative = sp.Matrix([sp.diff(objective, p) * p * (1 - p),
                                sp.diff(objective, q) * q * (1 - q)])
        for pv, qv, cv in [(Q(1, 4), Q(2, 3), Q(1, 7)),
                          (Q(3, 5), Q(1, 3), Q(2, 5)),
                          (Q(4, 5), Q(5, 6), Q(0))]:
            law = {p: pv, q: qv, c: cv}
            self.assertEqual(sp.simplify(weighted_score(probabilities, rewards, scores)
                                         - derivative).subs(law), sp.zeros(2, 1))
            # Different signs of mode gradient are a reward-design consequence.
            self.assertEqual(sp.sign(derivative[0].subs(law)), sp.sign(qv - cv - Q(1, 2)))

    def test_loo_for_varied_group_size_and_safe_equal_groups(self):
        for rewards in [[sp.Integer(0), sp.Integer(1)],
                        [Q(1, 3), Q(2, 3), Q(1)],
                        [Q(1)] * 5]:
            _, _, centered, standardized, loo = group_coefficients(rewards)
            k = len(rewards)
            for i in range(k):
                direct_loo = rewards[i] - sum(rewards[j] for j in range(k) if i != j) / sp.Integer(k - 1)
                self.assertEqual(loo[i], direct_loo)
            self.assertEqual(sum(centered), 0)
            if len(set(rewards)) == 1:
                self.assertEqual(standardized, sp.zeros(k, 1))
        with self.assertRaises(ValueError):
            group_coefficients([Q(1)])

    def test_support_rejection_and_common_zero(self):
        with self.assertRaisesRegex(ValueError, 'support'):
            importance_weights([Q(1, 2), Q(1, 2)], [Q(1), Q(0)])
        self.assertEqual(importance_weights([Q(1), Q(0)], [Q(1), Q(0)]), sp.Matrix([1, 0]))
        with self.assertRaises(ValueError):
            importance_weights([Q(1, 2)], [Q(1)])

    def test_cutoff_bootstrap_with_nonunit_discount(self):
        self.assertEqual(td_target(Q(1, 5), Q(3, 4), Q(9, 10), truncated=True), Q(7, 8))
        self.assertEqual(td_target(Q(1, 5), Q(3, 4), Q(9, 10), terminated=True), Q(1, 5))
        delta, adv = gae([Q(0), Q(0)], [Q(1, 4), Q(1, 2), Q(3, 4)], [False, False], lam=Q(1))
        self.assertEqual(adv, sp.Matrix([Q(1, 2), Q(1, 4)]))
        with self.assertRaisesRegex(ValueError, 'Split'):
            gae([Q(0), Q(1)], [Q(0)] * 3, [True, True])

    def test_fragment_weights_and_empty_inclusion(self):
        fragments = [[Q(1), Q(1)], [-Q(1)]]
        self.assertEqual(reductions(fragments)[1], Q(0))
        self.assertEqual(reductions(fragments)[2], Q(1, 3))
        with self.assertRaises(ValueError):
            reductions([[]])

    def test_local_ratio_leaves_changed_history_distribution(self):
        # The answer heads agree, so their local ratios are one.
        q = Q(3, 4)
        behavior_contribution = Q(1, 4) * q * (1 - q)
        target_contribution = Q(1, 2) * q * (1 - q)
        self.assertEqual(behavior_contribution, Q(3, 64))
        self.assertEqual(target_contribution, Q(3, 32))
        self.assertNotEqual(behavior_contribution, target_contribution)

    def test_discounted_shaping_can_change_variable_horizon_ranking(self):
        # Equal nonzero terminal potential does not give equal weighted boundaries.
        gamma = Q(1, 2)
        one_step_boundary = gamma * Q(1)
        two_step_boundary = gamma ** 2 * Q(1)
        self.assertNotEqual(one_step_boundary, two_step_boundary)
        self.assertGreater(Q(1) + Q(0), -Q(1, 10) + Q(0))
        self.assertLess(Q(1) + Q(0), -Q(1, 10) + Q(2))


if __name__ == '__main__':
    unittest.main()
