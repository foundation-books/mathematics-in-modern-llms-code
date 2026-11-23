"""Behavioral tests, including failures omitted by the happy-path fixtures."""

import unittest
import sympy as sp
from reference import (
    ALPHAS, BETAS, INDEX, N_vocab, MASK, RHO, STATES, WEIGHTS,
    check_stochastic, cost_ledger, cumulative, kl, mask_kernel,
    pair_laboratory, pair_product_reverse, posterior, reveal_weight,
    reverse_from_marginal, should_stop, uniform_kernel, unequal_mass, verify,
)

R = sp.Rational


class DiffusionTests(unittest.TestCase):
    def test_reference_all_fixtures(self):
        result = verify()
        self.assertEqual(result["unequal_three_round"], "37/150")

    def test_probability_conservation_rejects_bad_kernels(self):
        for matrix in (sp.Matrix([[1, 1], [0, 1]]),
                       sp.Matrix([[2, -1], [0, 1]]), sp.ones(1, 2)):
            with self.assertRaises(ValueError):
                check_stochastic(matrix)

    def test_noise_extremes_and_invalid_schedule(self):
        self.assertEqual(uniform_kernel(0), sp.eye(N_vocab))
        self.assertEqual(mask_kernel(1)[:, MASK], sp.ones(N_vocab + 1, 1))
        for function in (uniform_kernel, mask_kernel):
            with self.assertRaises(ValueError):
                function(R(3, 2))

    def test_posterior_support_is_not_epsilon_patched(self):
        matrices = [mask_kernel(b) for b in BETAS]
        products = cumulative(matrices)
        with self.assertRaises(ValueError):
            posterior(products[1], matrices[1], 0, 1)
        self.assertEqual(posterior(products[1], matrices[1], 0, 0),
                         sp.Matrix([[1, 0, 0, 0]]))

    def test_unsupported_reverse_rows_are_arbitrary_unused_extensions(self):
        prior = sp.Matrix([[1, 0]])
        reverse, support = reverse_from_marginal(prior, sp.eye(2))
        self.assertEqual(support, (True, False))
        changed = reverse.copy()
        changed[1, :] = sp.Matrix([[1, 0]])
        self.assertEqual(prior * changed, prior * reverse)

    def test_reveal_boundary_and_zero_kl_terms(self):
        self.assertIsNone(reveal_weight(1, 1))
        self.assertEqual(reveal_weight(R(1, 2), R(1, 2)), 0)
        self.assertEqual(reveal_weight(1, R(1, 2)), 1)
        with self.assertRaises(ValueError):
            reveal_weight(R(1, 4), R(1, 2))
        self.assertEqual(kl([0, 1], [0, 1]), 0)
        self.assertEqual(kl([R(1, 2), R(1, 2)], [0, 1]), sp.oo)

    def test_synchronous_update_does_not_read_a_new_neighbor(self):
        matrix = pair_product_reverse(1)
        self.assertEqual(matrix[INDEX[MASK, MASK], INDEX[0, 1]], RHO[0] * RHO[1])
        # A sequential read-after-write update would assign this event zero.
        self.assertGreater(matrix[INDEX[MASK, MASK], INDEX[0, 1]], 0)

    def test_copy_through_and_disagreement_extension(self):
        for w in WEIGHTS:
            matrix = pair_product_reverse(w)
            self.assertEqual(matrix[INDEX[0, 1], INDEX[0, 1]], 1)
            self.assertEqual(matrix[INDEX[0, MASK], INDEX[0, 1]], 0)

    def test_three_step_product_leaves_data_support(self):
        result = pair_laboratory()
        self.assertEqual(unequal_mass(result["exact_trace"][-1]), 0)
        self.assertEqual(unequal_mass(result["factorized_trace"][-1]), R(37, 150))
        self.assertTrue(all(result["marginals"][0][i] == 0
                            for i, state in enumerate(STATES) if state[0] != state[1]))

    def test_cost_break_even_is_strict(self):
        self.assertEqual(cost_ledger(R(11, 7))["ar_over_diffusion"], 1)
        self.assertGreater(cost_ledger(1)["ar_over_diffusion"], 1)
        self.assertLess(cost_ledger(2)["ar_over_diffusion"], 1)

    def test_commit_calls_change_ratio_denominator_but_cost_change_does_not(self):
        costly, cheaper = cost_ledger(2), cost_ledger(1)
        for ledger in (costly, cheaper):
            self.assertEqual(ledger["generation_calls"], 9)
            self.assertEqual(ledger["valid_tokens_per_generation_call"], R(4, 3))
            self.assertEqual(ledger["valid_tokens_per_denoising_evaluation"], R(12, 7))
            self.assertEqual(ledger["processed_positions"], 36)
        self.assertEqual((costly["diffusion_latency"], cheaper["diffusion_latency"]), (15, 8))

    def test_stopping_requires_stability_and_handles_cap(self):
        self.assertFalse(should_stop(1, (0,), None, 0, R(1, 100), 2))
        self.assertFalse(should_stop(2, (1,), (0,), 0, R(1, 100), 3))
        self.assertFalse(should_stop(2, (0,), (0,), 1, R(1, 100), 3))
        self.assertTrue(should_stop(2, (0,), (0,), 0, R(1, 100), 3))
        self.assertTrue(should_stop(1, (1,), None, 1, 0, 1))
        with self.assertRaises(ValueError):
            should_stop(1, (0,), None, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
