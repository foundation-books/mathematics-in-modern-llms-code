import unittest
import sympy as sp
import extensions_reference as fp

R = sp.Rational
# Literal format table is independent of the exponent-field decoder.
VALUES = (0, R(1, 2), 1, R(3, 2), 2, 3, 4, 6)


def cell_encoder(value, scale):
    """Independent decision-boundary oracle, rather than a nearest-code search."""
    mag = abs(value / scale)
    boundaries = (R(1, 4), R(3, 4), R(5, 4), R(7, 4), R(5, 2), R(7, 2), 5)
    selected = 7
    for j, boundary in enumerate(boundaries):
        if mag < boundary:
            selected = j
            break
        if mag == boundary:
            selected = j if j % 2 == 0 else j + 1
            break
    return selected | (8 if value < 0 and selected else 0)


class E2M1Tests(unittest.TestCase):
    def test_all_patterns(self):
        for code in range(16):
            self.assertEqual(fp.decode(code), (-1 if code >= 8 else 1) * VALUES[code % 8])
        self.assertEqual(len({fp.decode(code) for code in range(16)}), 15)
        self.assertEqual(fp.decode(0), fp.decode(8))
        self.assertNotEqual(0, 8)  # codes differ although real values coincide
        for invalid in [-1, 16, True, R(1, 2), '0001']:
            with self.assertRaises(ValueError):
                fp.decode(invalid)

    def test_all_signed_midpoints_and_cell_oracle(self):
        expected = [0, 1, 1, 2, 2, 4, 4]
        for midpoint, value in zip([R(1, 4), R(3, 4), R(5, 4), R(7, 4), R(5, 2), R(7, 2), 5], expected):
            for sign in [-1, 1]:
                self.assertEqual(fp.decode(fp.encode(sign * midpoint)), sign * value)
                self.assertEqual(fp.encode(sign * midpoint), cell_encoder(sign * midpoint, 1))
        for scale in [R(1, 4), R(1, 2), 1, R(7, 3)]:
            for numerator in range(-200, 201):
                value = R(numerator, 20)
                self.assertEqual(fp.encode(value, scale), cell_encoder(value, scale))
                code = fp.encode(value, scale)
                self.assertEqual(fp.reconstruct(value, scale), scale * fp.decode(code))
        self.assertEqual(fp.encode(R(-1, 4)), 0)
        # Ties-away would give3/2 instead of1 at5/4.
        self.assertNotEqual(fp.decode(fp.encode(R(5, 4))), R(3, 2))

    def test_zero_endpoints_saturation_validation(self):
        for value, expected in [(-100, -6), (-6, -6), (0, 0), (6, 6), (100, 6)]:
            self.assertEqual(fp.reconstruct(value, 1), expected)
        for bad in [sp.oo, -sp.oo, sp.nan, 0.2, sp.sqrt(2), True]:
            with self.assertRaises(ValueError):
                fp.encode(bad)
        for bad in [0, -1, sp.oo, 0.5]:
            with self.assertRaises(ValueError):
                fp.encode(1, bad)

    def test_interval_error_and_local_accuracy_reversals(self):
        for left, right in zip(VALUES, VALUES[1:]):
            for i in range(21):
                x = R(left) + (R(right) - left) * R(i, 20)
                self.assertLessEqual(abs(fp.reconstruct(x, 1) - x), (R(right) - left) / 2)
        self.assertGreater(abs(fp.reconstruct(10, 1) - 10), 1)
        def uniform(x):
            # enumerate the independent uniform levels and pick nearest for these untied witnesses
            return min([R(3 * j, 7) for j in range(-7, 8)], key=lambda y: abs(y - x))
        self.assertEqual(uniform(R(1, 5)), 0)
        self.assertEqual(uniform(R(6, 5)), R(9, 7))
        self.assertLess(abs(fp.reconstruct(R(1, 5), R(1, 2)) - R(1, 5)), abs(uniform(R(1, 5)) - R(1, 5)))
        self.assertLess(abs(uniform(R(6, 5)) - R(6, 5)), abs(fp.reconstruct(R(6, 5), R(1, 2)) - R(6, 5)))
        self.assertNotEqual(fp.reconstruct(R(1, 5), R(1, 2)), uniform(R(1, 5)))

    def test_complete_forward_master_update_and_requantization(self):
        masters = [R(1, 5), R(7, 10), R(11, 10), R(14, 5)]
        got = fp.surrogate_update(masters, [1] * 4, 4, R(1, 2), R(1, 10))
        self.assertEqual(sum(masters), R(24, 5))
        expected = dict(codes=(1, 3, 4, 7), payload=(R(1, 2), R(3, 2), 2, 6),
                        weights=(R(1, 4), R(3, 4), 1, 3), y=5, loss=R(1, 2),
                        mask=(1, 1, 1, 1), gradient=(1, 1, 1, 1),
                        new_masters=(R(1, 10), R(3, 5), 1, R(27, 10)),
                        new_codes=(0, 2, 4, 7), new_weights=(0, R(1, 2), 1, 3),
                        new_y=R(9, 2), new_loss=R(1, 8))
        self.assertEqual(got, expected)
        # Independent smooth loss in reconstructed variables supplies the incoming gradient.
        q = sp.symbols('q:4', real=True)
        loss = (sum(q) - 4) ** 2 / 2
        grad = tuple(sp.diff(loss, v).subs(dict(zip(q, got['weights']))) for v in q)
        self.assertEqual(grad, got['gradient'])
        for i, w in enumerate(masters):
            for delta in [R(1, 10000), R(-1, 10000)]:
                self.assertEqual(fp.reconstruct(w + delta, R(1, 2)), got['weights'][i])
        wrong_codes = [R(c) - R(1, 10) for c in got['codes']]
        self.assertNotEqual(tuple(wrong_codes), got['new_codes'])
        with self.assertRaises(ValueError):
            fp.decode(wrong_codes[0])

    def test_surrogate_endpoints_and_nonunit_inputs(self):
        got = fp.surrogate_update([-4, -3, 0, 3, 4], [1] * 5, 1, R(1, 2), R(1, 10))
        self.assertEqual(got['mask'], (0, 0, 1, 0, 0))
        self.assertEqual(got['gradient'], (0, 0, -1, 0, 0))
        got = fp.surrogate_update([1, 1], [2, -1], 0, 1, R(1, 10))
        self.assertEqual(got['gradient'], (2, -1))

    def test_group_scales_padding_and_storage(self):
        self.assertEqual(fp.group_scaled_dot([(R(1, 2), [1, 1]), (2, [1, 1])]), 5)
        self.assertNotEqual(4 * R(1, 2), 5)
        base = fp.surrogate_update([R(1, 5), R(7, 10), R(11, 10), R(14, 5)], [1] * 4, 4, R(1, 2), R(1, 10))
        padded = fp.surrogate_update([R(1, 5), R(7, 10), R(11, 10), R(14, 5)] + [0] * 28,
                                      [1] * 4 + [0] * 28, 4, R(1, 2), R(1, 10))
        self.assertEqual(base['y'], padded['y'])
        self.assertEqual(base['new_loss'], padded['new_loss'])
        self.assertEqual(padded['gradient'][4:], (0,) * 28)
        self.assertEqual((32 * 4 + 8, R(32 * 4 + 8, 8)), (136, 17))


if __name__ == '__main__':
    unittest.main()
