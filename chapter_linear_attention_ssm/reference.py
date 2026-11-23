"""Instructor reference for Chapter 8; exact correctness checks, not benchmarks.

Domain: nonempty finite sequences of exact real SymPy column matrices.
Attention query/key coordinates must be provably nonnegative and each read
denominator provably positive. Incoming (S,n) provenance is a caller contract.
The expanded and pairwise routines are independent of the recurrent routines.
"""
import sys
import sympy as sp
from student_setup import fixture, rectangular_values, ssm_fixture, affine_fixture, exact_equal


def _matrix(value, name):
    if not isinstance(value, sp.MatrixBase) or min(value.shape) == 0:
        raise ValueError(f'{name} must be a nonempty SymPy matrix.')
    if any(x.has(sp.Float) or x.is_real is not True or x.is_finite is not True
           for x in value):
        raise ValueError(f'{name} must contain exact finite real expressions.')
    return sp.Matrix(value)


def _features(queries, keys, values):
    if not keys or not (len(queries) == len(keys) == len(values)):
        raise ValueError('Supply equally long, nonempty sequences.')
    first_k = _matrix(keys[0], 'key')
    first_v = _matrix(values[0], 'value')
    r, d_v = first_k.rows, first_v.rows
    for q, k, v in zip(queries, keys, values):
        q, k, v = (_matrix(q, 'query'), _matrix(k, 'key'), _matrix(v, 'value'))
        if q.shape != (r, 1) or k.shape != (r, 1) or v.shape != (d_v, 1):
            raise ValueError('Supply columns of the declared feature/value widths.')
        if any(x.is_nonnegative is not True for vec in (q, k) for x in vec):
            raise ValueError('Query/key features must be provably nonnegative.')
    return r, d_v


def _read(S, normalizer_state, q):
    denominator = sp.simplify((normalizer_state.T * q)[0])
    if denominator.is_positive is not True:
        raise ValueError('A read requires a provably positive denominator.')
    return (S * q / denominator).applyfunc(sp.simplify)


def pairwise_feature_attention(query_features, key_features, values):
    """Return all inclusive outputs, using independent explicit pair sums."""
    _, d_v = _features(query_features, key_features, values)
    outputs = []
    for t, query in enumerate(query_features):
        weights = [(query.T * key_features[j])[0] for j in range(t + 1)]
        denominator = sp.simplify(sum(weights, sp.S.Zero))
        if denominator.is_positive is not True:
            raise ValueError('A read requires a provably positive denominator.')
        numerator = sum((weights[j] * values[j] for j in range(t + 1)),
                        sp.zeros(d_v, 1))
        outputs.append((numerator / denominator).applyfunc(sp.simplify))
    return outputs


def recurrent_feature_attention(query_features, key_features, values, S0=None, normalizer_state0=None):
    """Return (outputs, prefix (S,n) states); write current token before read."""
    r, d_v = _features(query_features, key_features, values)
    S = sp.zeros(d_v, r) if S0 is None else _matrix(S0, 'S0')
    normalizer_state = sp.zeros(r, 1) if normalizer_state0 is None else _matrix(normalizer_state0, 'normalizer_state0')
    if S.shape != (d_v, r) or normalizer_state.shape != (r, 1):
        raise ValueError('Incoming state shapes do not match.')
    if any(x.is_nonnegative is not True for x in normalizer_state):
        raise ValueError('Incoming normalizer state must be nonnegative.')
    outputs, states = [], []
    for q, k, v in zip(query_features, key_features, values):
        S, normalizer_state = S + v * k.T, normalizer_state + k
        outputs.append(_read(S, normalizer_state, q))
        states.append((S.copy(), normalizer_state.copy()))
    return outputs, states


def decayed_feature_attention(query_features, key_features, values, retention):
    """Return (outputs, states); retain both states and allow zero gates."""
    r, d_v = _features(query_features, key_features, values)
    if len(retention) != len(values):
        raise ValueError('Supply one retention factor per position.')
    factors = [sp.sympify(x) for x in retention]
    for factor in factors:
        if (factor.has(sp.Float) or factor.is_real is not True
                or factor.is_nonnegative is not True
                or (1 - factor).is_nonnegative is not True):
            raise ValueError('Retention must be exact and provably in [0,1].')
    S, normalizer_state = sp.zeros(d_v, r), sp.zeros(r, 1)
    outputs, states = [], []
    for q, k, v, factor in zip(query_features, key_features, values, factors):
        S, normalizer_state = factor * S + v * k.T, factor * normalizer_state + k
        outputs.append(_read(S, normalizer_state, q))
        states.append((S.copy(), normalizer_state.copy()))
    return outputs, states


def _ssm_inputs(inputs, A, B, C, D, h0):
    A, B, C, D, h0 = [_matrix(x, name) for x, name in
                       zip((A, B, C, D, h0), ('A', 'B', 'C', 'D', 'h0'))]
    n, p, m = A.rows, B.cols, C.rows
    if (A.shape != (n, n) or B.shape != (n, p)
            or C.shape != (m, n) or D.shape != (m, p) or h0.shape != (n, 1)):
        raise ValueError('SSM matrices or initial state have incompatible shapes.')
    if not inputs:
        raise ValueError('Supply a nonempty input sequence.')
    for u in inputs:
        if _matrix(u, 'input').shape != (p, 1):
            raise ValueError('SSM inputs must be columns of width p.')
    return A, B, C, D, h0


def ssm_recurrence(inputs, A, B, C, D, h0):
    """Return (outputs, states) using the inclusive state-space recurrence."""
    A, B, C, D, h = _ssm_inputs(inputs, A, B, C, D, h0)
    outputs, states = [], []
    for u in inputs:
        h = A * h + B * u
        outputs.append((C * h + D * u).applyfunc(sp.simplify))
        states.append(h.copy())
    return outputs, states


def ssm_expanded(inputs, A, B, C, D, h0):
    """Return outputs from the independent lag sum, including initial response."""
    A, B, C, D, h0 = _ssm_inputs(inputs, A, B, C, D, h0)
    outputs = []
    for t in range(1, len(inputs) + 1):
        output = C * A**t * h0 + D * inputs[t - 1]
        output += sum((C * A**(t - j) * B * inputs[j - 1]
                       for j in range(1, t + 1)), sp.zeros(C.rows, 1))
        outputs.append(output.applyfunc(sp.simplify))
    return outputs


def _pair(pair):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError('Supply an (A,b) affine pair.')
    A, b = _matrix(pair[0], 'transition'), _matrix(pair[1], 'write')
    if A.rows != A.cols or b.shape != (A.rows, 1):
        raise ValueError('An affine pair needs a square transition and column write.')
    return A, b


def compose_affine(earlier_pair, later_pair):
    """Chronological composition: the left pair acts before the right pair."""
    A_a, b_a = _pair(earlier_pair)
    A_b, b_b = _pair(later_pair)
    if A_a.shape != A_b.shape:
        raise ValueError('Affine segments must have the same state width.')
    return A_b * A_a, A_b * b_a + b_b


def affine_prefix_reference(pairs, h0):
    """Return (states, prefix pairs); this is a serial correctness reference."""
    if not pairs:
        raise ValueError('Supply a nonempty pair sequence.')
    h0 = _matrix(h0, 'h0')
    n = _pair(pairs[0])[0].rows
    if h0.shape != (n, 1):
        raise ValueError('The incoming state must match the affine pairs.')
    prefix = (sp.eye(n), sp.zeros(n, 1))
    states, prefixes = [], []
    for pair in pairs:
        prefix = compose_affine(prefix, pair)
        prefixes.append((prefix[0].copy(), prefix[1].copy()))
        states.append((prefix[0] * h0 + prefix[1]).applyfunc(sp.simplify))
    return states, prefixes


def balanced_reduce(pairs):
    """Return one segment pair with balanced parentheses and unchanged order.

    Repeated calls over tiny prefixes below are checks, not a parallel scan.
    """
    if not pairs:
        raise ValueError('Supply a nonempty segment.')
    validated = [_pair(pair) for pair in pairs]
    if len({A.shape for A, _ in validated}) != 1:
        raise ValueError('Affine segments must have the same state width.')

    def reduce_segment(items):
        if len(items) == 1:
            return items[0]
        middle = len(items) // 2
        return compose_affine(reduce_segment(items[:middle]),
                              reduce_segment(items[middle:]))

    return reduce_segment(validated)


def _same(actual, expected):
    assert exact_equal(actual, expected), (actual, expected)


def _same_outputs(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        _same(a, b)


def run_checks():
    q, k, v = fixture()
    direct = pairwise_feature_attention(q, k, v)
    recurrent, states = recurrent_feature_attention(q, k, v)
    expected = [sp.Matrix([1, 0]), sp.Matrix([sp.Rational(1, 3), sp.Rational(2, 3)]),
                sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 6)])]
    _same_outputs(direct, recurrent)
    _same_outputs(recurrent, expected)
    _same(states[-1][0], sp.Matrix([[2, 1], [1, 2]]))
    _same(states[-1][1], sp.Matrix([2, 2]))
    continued, _ = recurrent_feature_attention(q[1:], k[1:], v[1:], *states[0])
    _same_outputs(continued, recurrent[1:])
    changed, _ = recurrent_feature_attention(q, k, v[:2] + [sp.Matrix([9, -4])])
    _same_outputs(changed[:2], recurrent[:2])
    vr = rectangular_values()
    rectangular, rst = recurrent_feature_attention(q, k, vr)
    _same_outputs(rectangular, pairwise_feature_attention(q, k, vr))
    assert all(S.shape == (3, 2) for S, _ in rst)
    _same(rst[-1][0], sp.Matrix([[2, 1], [1, 2], [5, 2]]))
    _same(rectangular[-1], sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 6), sp.Rational(3, 2)]))
    for index, ((S, normalizer_state), output) in enumerate(zip(states, recurrent), start=1):
        print('Attention prefix', index, '| S =', S.tolist(), '| n =', list(normalizer_state),
              '| numerator =', list(S * q[index - 1]),
              '| Z =', (normalizer_state.T * q[index - 1])[0], '| output =', list(output))

    repeat = [k[0], k[0]]
    first, st1 = recurrent_feature_attention(repeat, repeat, v[:2])
    second, st2 = recurrent_feature_attention(repeat, repeat, list(reversed(v[:2])))
    for a, b in zip(st1[-1], st2[-1]):
        _same(a, b)
    _same(first[-1], sp.ones(2, 1) / 2)
    _same(first[-1], second[-1])
    K = sp.ones(3)
    L = sp.Matrix([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    P = sp.diag(1, sp.Rational(1, 2), sp.Rational(1, 3)) * L
    assert (K.rank(), L.rank(), P.rank()) == (1, 3, 3)
    _same(P.det(), sp.Rational(1, 6))
    # This scalar comparison concerns only the admitted two-token read.
    # Its first key is zero, so a streaming first-prefix read is undefined.
    scalar_similarity = [sp.Integer(0), sp.Integer(1)]
    scalar_values = [sp.Integer(0), sp.Integer(1)]
    scalar_read = sum(w * vj for w, vj in zip(scalar_similarity, scalar_values)) / sum(scalar_similarity)
    _same(scalar_read, 1)

    decayed, dst = decayed_feature_attention(q, k, v, [sp.Rational(1, 2)] * 3)
    _same(dst[-1][0], sp.Matrix([[sp.Rational(5, 4), 1], [1, sp.Rational(3, 2)]]))
    _same(dst[-1][1], sp.Matrix([sp.Rational(5, 4), sp.Rational(3, 2)]))
    _same(decayed[-1], sp.Matrix([sp.Rational(13, 17), sp.Rational(16, 17)]))
    reset, _ = decayed_feature_attention(q, k, v, [1, 1, 0])
    _same(reset[-1], v[-1])
    assert sp.simplify(1 - sp.E / (1 + sp.E)).is_positive is True

    A, B, C, D = ssm_fixture()
    for h0 in (sp.zeros(2, 1), sp.Matrix([2, -1])):
        for feedthrough in (D, sp.diag(1, 2)):
            output, _ = ssm_recurrence(v, A, B, C, feedthrough, h0)
            _same_outputs(output, ssm_expanded(v, A, B, C, feedthrough, h0))
    _same(ssm_expanded(v, A, B, C, D, sp.zeros(2, 1))[-1],
          sp.Matrix([sp.Rational(5, 4), sp.Rational(4, 3)]))
    _same(ssm_expanded(v, A, B, C, D, sp.Matrix([2, -1]))[-1],
          sp.Matrix([sp.Rational(3, 2), sp.Rational(35, 27)]))
    alpha, delta, time = sp.symbols('alpha delta time', positive=True)
    integral = sp.integrate(sp.exp(-alpha * time), (time, 0, delta))
    _same(integral, (1 - sp.exp(-alpha * delta)) / alpha)
    _same(sp.limit(integral, alpha, 0, dir='+'), delta)
    for rate in (sp.log(2), sp.log(3)):
        gain = rate / (1 - sp.exp(-rate))
        _same(gain * integral.subs({alpha: rate, delta: 1}), 1)

    pairs = affine_fixture()
    assert pairs[0][0] * pairs[1][0] != pairs[1][0] * pairs[0][0]
    left = compose_affine(compose_affine(pairs[0], pairs[1]), pairs[2])
    right = compose_affine(pairs[0], compose_affine(pairs[1], pairs[2]))
    for a, b in zip(left, right):
        _same(a, b)
    for items in (pairs, pairs[:1] + [(sp.zeros(2), sp.Matrix([4, 5]))] + pairs[2:]):
        h0 = sp.Matrix([2, -1])
        h = h0.copy()
        prefix_states, _ = affine_prefix_reference(items, h0)
        for t, (At, bt) in enumerate(items, start=1):
            h = At * h + bt
            _same(prefix_states[t - 1], h)
            Ap, bp = balanced_reduce(items[:t])
            _same(Ap * h0 + bp, h)
        after_one = prefix_states[0]
        continued, _ = affine_prefix_reference(items[1:], after_one)
        _same_outputs(continued, prefix_states[1:])
    scalar_pairs = [(sp.Matrix([[a]]), sp.Matrix([b])) for a, b in ((2, 1), (3, 4), (5, 6))]
    As, bs = balanced_reduce(scalar_pairs)
    _same(As, sp.Matrix([[30]]))
    _same(bs, sp.Matrix([41]))
    _same(As * sp.Matrix([2]) + bs, sp.Matrix([101]))
    _same(balanced_reduce(list(reversed(scalar_pairs)))[1], sp.Matrix([45]))

    epsilon, Z = sp.symbols('epsilon Z', positive=True)
    _same(Z / (Z + epsilon) - 1, -epsilon / (Z + epsilon))
    assert states[-1][0] * q[-1] != expected[-1]
    try:
        recurrent_feature_attention([sp.Matrix([0, 1])], [k[0]], [v[0]])
    except ValueError:
        pass
    else:
        raise AssertionError('Zero read must be rejected.')
    for args in (([], [], []), (q, k[:2], v), ([sp.ones(3, 1)], [k[0]], [v[0]])):
        try:
            recurrent_feature_attention(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong lengths/shapes must be rejected.')
    assert 256 * (64 + 64) == 32768 and 64 * 64 + 64 == 4160
    print('All exact Chapter 8 reference checks passed.')
    print('Python:', sys.version.split()[0], '| SymPy:', sp.__version__)


if __name__ == '__main__':
    run_checks()
