"""Finite E2M1 values and a separately declared surrogate update.

Not a hardware emulator: masters, reconstructed products and sums are exact.
"""
import sympy as sp

R = sp.Rational
MAGNITUDES = (sp.S.Zero, R(1, 2), sp.S.One, R(3, 2), sp.Integer(2),
              sp.Integer(3), sp.Integer(4), sp.Integer(6))


def exact(value):
    if isinstance(value, (bool, float)):
        raise ValueError('finite exact rational required')
    value = sp.sympify(value)
    if value.is_Rational is not True:
        raise ValueError('finite exact rational required')
    return value


def scale_value(scale):
    scale = exact(scale)
    if scale <= 0:
        raise ValueError('positive scale required')
    return scale


def decode(code):
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError('four-bit code required')
    sign = -1 if code & 8 else 1
    exponent, fraction = (code >> 1) & 3, code & 1
    if exponent == 0:
        return sign * R(fraction, 2)
    return sign * (1 + R(fraction, 2)) * sp.Integer(2) ** (exponent - 1)


def encode(value, scale=1):
    value, scale = exact(value), scale_value(scale)
    normalized = abs(value / scale)
    # The fraction bit is the low bit of each three-bit magnitude encoding.
    magnitude_code = min(range(8), key=lambda c: (abs(decode(c) - normalized), c & 1, c))
    return magnitude_code | (8 if value < 0 and magnitude_code != 0 else 0)


def reconstruct(value, scale):
    scale = scale_value(scale)
    return scale * decode(encode(value, scale))


def surrogate_update(masters, inputs, target, scale, rate):
    masters = tuple(exact(v) for v in masters)
    inputs = tuple(exact(v) for v in inputs)
    target, scale, rate = exact(target), scale_value(scale), exact(rate)
    if not masters or len(masters) != len(inputs) or rate < 0:
        raise ValueError('matching nonempty readout and nonnegative learning rate required')
    codes = tuple(encode(w, scale) for w in masters)
    payload = tuple(decode(c) for c in codes)
    weights = tuple(scale * p for p in payload)
    y = sum((w * x for w, x in zip(weights, inputs)), sp.S.Zero)
    residual = y - target
    mask = tuple((1 if -6 * scale < w < 6 * scale else 0) for w in masters)
    gradient = tuple(residual * x * m for x, m in zip(inputs, mask))
    new_masters = tuple(w - rate * g for w, g in zip(masters, gradient))
    new_codes = tuple(encode(w, scale) for w in new_masters)
    new_weights = tuple(scale * decode(c) for c in new_codes)
    new_y = sum((w * x for w, x in zip(new_weights, inputs)), sp.S.Zero)
    return dict(codes=codes, payload=payload, weights=weights, y=y,
                loss=residual ** 2 / 2, mask=mask, gradient=gradient,
                new_masters=new_masters, new_codes=new_codes,
                new_weights=new_weights, new_y=new_y,
                new_loss=(new_y - target) ** 2 / 2)


def group_scaled_dot(groups):
    """Each group is (scale product, sequence of unscaled product values)."""
    return sum((scale_value(scale) * sum((exact(p) for p in products), sp.S.Zero)
                for scale, products in groups), sp.S.Zero)


if __name__ == '__main__':
    result = surrogate_update([R(1, 5), R(7, 10), R(11, 10), R(14, 5)],
                              [1] * 4, 4, R(1, 2), R(1, 10))
    print('E2M1 magnitudes:', MAGNITUDES)
    for key, value in result.items():
        print(key, value)
    print('Payload/scale block bits:', 32 * 4 + 8)
    print('Exact constructed numerical path; no complete hardware-format emulation.')
