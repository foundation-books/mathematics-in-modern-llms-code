"""Behavioral checks of Chapter 8 operators and their explicit domains."""
import unittest
import sympy as sp
import reference as ref
from student_setup import fixture, rectangular_values, ssm_fixture, affine_fixture, exact_equal


class ReferenceTests(unittest.TestCase):
    def assert_outputs_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        self.assertTrue(all(exact_equal(a, b) for a, b in zip(actual, expected)))

    def test_every_prefix_and_rectangular_values(self):
        q, k, v = fixture()
        for values in (v, rectangular_values()):
            with self.subTest(value_width=values[0].rows):
                direct = ref.pairwise_feature_attention(q, k, values)
                stream, states = ref.recurrent_feature_attention(q, k, values)
                self.assert_outputs_equal(stream, direct)
                self.assertTrue(all(S.shape == (values[0].rows, 2) for S, _ in states))
        x, y = sp.symbols('x y', positive=True)
        values = [sp.Matrix([x, -y]), sp.Matrix([y, x])]
        queries = [sp.Matrix([x, 1]), sp.Matrix([1, y])]
        keys = [sp.Matrix([1, x]), sp.Matrix([y, 1])]
        self.assert_outputs_equal(ref.pairwise_feature_attention(queries, keys, values),
                                  ref.recurrent_feature_attention(queries, keys, values)[0])

    def test_continuation_and_future_exclusion(self):
        q, k, v = fixture()
        full, states = ref.recurrent_feature_attention(q, k, v)
        S0, normalizer_state0 = (x.copy() for x in states[0])
        tail, _ = ref.recurrent_feature_attention(q[1:], k[1:], v[1:], S0, normalizer_state0)
        self.assert_outputs_equal(tail, full[1:])
        self.assertTrue(exact_equal(S0, states[0][0]))
        self.assertTrue(exact_equal(normalizer_state0, states[0][1]))
        reset, _ = ref.recurrent_feature_attention(q[1:], k[1:], v[1:])
        self.assertFalse(exact_equal(reset[0], full[1]))
        changed, _ = ref.recurrent_feature_attention(q, k, v[:2] + [sp.Matrix([9, -4])])
        self.assert_outputs_equal(changed[:2], full[:2])

    def test_attention_rejections(self):
        q, k, v = fixture()
        cases = [([], [], []), (q, k[:1], v), ([sp.ones(3, 1)], [k[0]], [v[0]]),
                 ([q[0].T], [k[0]], [v[0]]), ([sp.Matrix([0, 1])], [k[0]], [v[0]]),
                 ([sp.Matrix([-1, 2])], [k[0]], [v[0]]),
                 ([sp.Matrix([0.5, 2])], [k[0]], [v[0]]),
                 ([q[0]], [[1, 0]], [v[0]]),
                 ([sp.Matrix([sp.Symbol('x', real=True), 2])], [k[0]], [v[0]])]
        for routine in (ref.pairwise_feature_attention, ref.recurrent_feature_attention):
            for args in cases:
                with self.subTest(routine=routine.__name__, inputs=args):
                    with self.assertRaises(ValueError):
                        routine(*args)
        with self.assertRaises(ValueError):
            ref.recurrent_feature_attention(q, k, v, sp.zeros(3, 2), sp.zeros(2, 1))
        with self.assertRaises(ValueError):
            ref.recurrent_feature_attention(q, k, v, normalizer_state0=sp.Matrix([-1, 0]))

    def test_decay_and_gate_domain(self):
        q, k, v = fixture()
        out, _ = ref.decayed_feature_attention(q, k, v, [sp.Rational(1, 2)] * 3)
        self.assertTrue(exact_equal(out[-1], sp.Matrix([sp.Rational(13, 17), sp.Rational(16, 17)])))
        reset, _ = ref.decayed_feature_attention(q, k, v, [1, 1, 0])
        self.assertTrue(exact_equal(reset[-1], v[-1]))
        for gates in ([1], [1, 1, 2], [1, 1, -1], [1, 1, 0.5]):
            with self.subTest(gates=gates):
                with self.assertRaises(ValueError):
                    ref.decayed_feature_attention(q, k, v, gates)

    def test_ssm_initial_state_feedthrough_and_shapes(self):
        _, _, v = fixture()
        A, B, C, D = ssm_fixture()
        h0 = sp.Matrix([2, -1])
        for direct in (D, sp.diag(1, 2)):
            out, _ = ref.ssm_recurrence(v, A, B, C, direct, h0)
            self.assert_outputs_equal(out, ref.ssm_expanded(v, A, B, C, direct, h0))
        base = ref.ssm_expanded(v, A, B, C, D, h0)
        fed = ref.ssm_expanded(v, A, B, C, sp.diag(1, 2), h0)
        self.assert_outputs_equal([f - b for f, b in zip(fed, base)],
                                  [sp.diag(1, 2) * u for u in v])
        for routine in (ref.ssm_recurrence, ref.ssm_expanded):
            for args in (([], A, B, C, D, h0), (v, A, sp.ones(3, 2), C, D, h0),
                         (v, A, B, C, sp.ones(3, 2), h0), (v, A, B, C, D, h0.T),
                         ([sp.ones(3, 1)], A, B, C, D, h0)):
                with self.subTest(routine=routine.__name__):
                    with self.assertRaises(ValueError):
                        routine(*args)

    def test_noncommuting_prefixes_and_zero_transition(self):
        pairs, h0 = affine_fixture(), sp.Matrix([2, -1])
        states, _ = ref.affine_prefix_reference(pairs, h0)
        self.assert_outputs_equal(states, [sp.Matrix([2, -1]), sp.Matrix([2, 2]), sp.Matrix([5, 7])])
        for t, state in enumerate(states, start=1):
            A, b = ref.balanced_reduce(pairs[:t])
            self.assertTrue(exact_equal(A * h0 + b, state))
        identity = (sp.eye(2), sp.zeros(2, 1))
        for combined in (ref.compose_affine(identity, pairs[0]), ref.compose_affine(pairs[0], identity)):
            self.assertTrue(all(exact_equal(a, b) for a, b in zip(combined, pairs[0])))
        reset_pairs = pairs[:1] + [(sp.zeros(2), sp.Matrix([4, 5]))] + pairs[2:]
        reset, _ = ref.affine_prefix_reference(reset_pairs, h0)
        self.assert_outputs_equal(reset, [sp.Matrix([2, -1]), sp.Matrix([4, 5]), sp.Matrix([9, 16])])
        _, write = ref.balanced_reduce(reset_pairs)
        self.assertTrue(exact_equal(write, reset[-1]))
        for routine, args in ((ref.balanced_reduce, ([],)),
                              (ref.affine_prefix_reference, (pairs, sp.zeros(3, 1))),
                              (ref.compose_affine, (pairs[0], (sp.eye(3), sp.zeros(3, 1)))),
                              (ref.compose_affine, ((sp.ones(2, 3), sp.zeros(2, 1)), pairs[0]))):
            with self.subTest(routine=routine.__name__):
                with self.assertRaises(ValueError):
                    routine(*args)


if __name__ == '__main__':
    unittest.main()
