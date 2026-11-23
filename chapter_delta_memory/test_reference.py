"""Behavioral checks for the mathematical API and causal boundaries."""
import unittest
import sympy as sp
from reference import (step, step_pullback, affine_chunks, triangular_chunk,
                       triangular_chunks, five_event_fixture, sequential)
from student_setup import exact_equal


class CorrectiveMemoryTests(unittest.TestCase):
    def test_rectangular_shapes_and_invalid_inputs(self):
        S, k, v = sp.zeros(3, 2), sp.Matrix([1, 0]), sp.Matrix([1, 2, 3])
        self.assertTrue(exact_equal(step(S, k, v, k)[4], v))
        with self.assertRaisesRegex(ValueError, "value shape"):
            step(S, k, sp.Matrix([1, 2]), k)
        with self.assertRaisesRegex(ValueError, "query shape"):
            step(S, k, v, k.T)
        with self.assertRaisesRegex(ValueError, "diagonal"):
            step(S, k, v, k, D=sp.Matrix([[1, 1], [0, 1]]))

    def test_retention_precedes_prediction(self):
        S, k, v = sp.eye(2), sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)]), sp.Matrix([0, 1])
        D = sp.diag(sp.Rational(1, 2), sp.Rational(3, 4))
        retained, prediction, error, updated, _ = step(S, k, v, k, D=D)
        self.assertTrue(exact_equal(prediction, retained * k))
        self.assertTrue(exact_equal(updated, retained + error * k.T))
        wrong = S * (sp.eye(2) - k * k.T) * D + v * k.T
        self.assertFalse(exact_equal(updated, wrong))

    def test_prefix_outputs_exclude_future_values(self):
        initial, ks, vs, qs, bs, ds = five_event_fixture()
        old = affine_chunks(initial, ks, vs, qs, bs, ds, [5])
        changed = list(vs)
        changed[-1] = sp.Matrix([100, 200, 300])
        new = affine_chunks(initial, ks, changed, qs, bs, ds, [5])
        for before, after in zip(old[1][:-1], new[1][:-1]):
            self.assertTrue(exact_equal(before, after))
        self.assertFalse(exact_equal(old[1][-1], new[1][-1]))

    def test_triangular_zero_beta_and_nonzero_incoming_state(self):
        initial, ks, vs, qs, bs, _ = five_event_fixture()
        states, outputs = sequential(initial, ks, vs, qs, bs, [sp.eye(2)] * 5)
        final, computed, _, U = triangular_chunk(initial, ks, vs, qs, bs)
        self.assertTrue(exact_equal(U.row(2), sp.zeros(1, 3)))
        self.assertTrue(exact_equal(final, states[-1]))
        for a, b in zip(computed, outputs):
            self.assertTrue(exact_equal(a, b))

    def test_gated_triangular_short_tail_and_zero_coordinate(self):
        initial, ks, vs, qs, bs, ds = five_event_fixture()
        states, outputs = sequential(initial, ks, vs, qs, bs, ds)
        final, computed = triangular_chunks(initial, ks, vs, qs, bs, [2, 2, 1], ds)
        self.assertTrue(exact_equal(final, states[-1]))
        for actual, expected in zip(computed, outputs):
            self.assertTrue(exact_equal(actual, expected))

    def test_erased_state_column_can_have_a_gate_gradient(self):
        state = sp.Matrix([[3, 4]])
        key = sp.Matrix([1, 0])
        value = sp.Matrix([2])
        state_bar, gate_bar, *_ = step_pullback(
            state, key, value, key, sp.Rational(1, 2), sp.diag(0, 1),
            sp.Matrix([1]), sp.zeros(1, 2),
        )
        self.assertEqual(state_bar, sp.zeros(1, 2))
        self.assertEqual(gate_bar, sp.Matrix([sp.Rational(3, 2), 0]))

    def test_partition_validation(self):
        fixture = five_event_fixture()
        for partition in ([2, 2], [2, 0, 3], [2, -1, 4]):
            with self.assertRaises(ValueError):
                affine_chunks(*fixture, partition)


if __name__ == "__main__":
    unittest.main()
