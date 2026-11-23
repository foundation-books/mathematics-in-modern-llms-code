"""Behavioral regressions for domains, causal dependencies and false equivalences."""

from dataclasses import replace
import unittest

import sympy as sp

from fixture import assert_zero, fixture, softmax_exact
from reference import absorbed, compare, expanded, grouped


class AttentionStateTests(unittest.TestCase):
    def test_rejects_invalid_interfaces(self):
        data = fixture()
        for scores in ([], sp.Matrix([[1, 2]])):
            with self.assertRaises(ValueError):
                softmax_exact(scores)
        for t in (0, 4, True, sp.Rational(3, 2)):
            with self.assertRaises(ValueError):
                expanded(data, t)
        for groups in ((0, 0, 0, 0), (0, 0, 2, 2), (0, 1), (False, 0, 1, 1)):
            with self.assertRaises(ValueError):
                grouped(data, 2, groups, (0, 2))
        bad = replace(data, UV=(sp.zeros(2, 2),)+data.UV[1:])
        with self.assertRaises(ValueError):
            absorbed(bad, 2)

    def test_future_input_is_excluded(self):
        data = fixture()
        altered = data.X.copy()
        altered[:, 2] = sp.Matrix([100, -7, 11, 13])
        changed = replace(data, X=altered)
        readers = [expanded, absorbed,
                   lambda f, t: grouped(f, t, (0, 0, 1, 1), (0, 2)),
                   lambda f, t: expanded(f, t, positional=True),
                   lambda f, t: absorbed(f, t, positional=True)]
        for t in (1, 2):
            for reader in readers:
                compare(reader(data, t), reader(changed, t))

    def test_wrong_latent_scale_changes_logits(self):
        data = fixture()
        C = data.D*data.X
        q = data.WQ[0]*data.X[:, 2]
        wrong_scores = C.T*data.UK[0].T*q/sp.sqrt(data.D.rows)
        with self.assertRaises(AssertionError):
            assert_zero(wrong_scores-expanded(data, 3).heads[0].scores)

    def test_one_shared_mixture_changes_an_unequal_head(self):
        data = fixture()
        correct = absorbed(data, 3)
        wrong_head2 = data.UV[1]*correct.heads[0].mixture
        with self.assertRaises(AssertionError):
            assert_zero(wrong_head2-correct.heads[1].output)

    def test_used_groups_may_have_unequal_numbers_of_query_heads(self):
        data = fixture()
        for t in (1, 2, 3):
            compare(grouped(data, t, (0, 0, 0, 1), (0, 2)),
                    expanded(data, t, (0, 0, 0, 2)))

    def test_general_factorization_on_distinct_inputs(self):
        data = fixture()
        X = sp.Matrix([[1, 2, -1], [2, -1, 3], [-2, 1, 2], [1, 2, 3]])
        changed = replace(data, X=X)
        for t in (1, 2, 3):
            compare(expanded(changed, t), absorbed(changed, t))


if __name__ == "__main__":
    unittest.main()
