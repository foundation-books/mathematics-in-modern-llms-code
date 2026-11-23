"""Exact attention calculations and independent Chapter 5 homework checks.

Run from the repository root with Python 3.10+ and SymPy. This is a symbolic
teaching implementation, not a production attention kernel or GPU benchmark.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
import json
import sys

import mpmath as mp
import sympy as sp


def fixture():
    scores = (sp.Integer(0), sp.log(2), sp.log(3))
    values = sp.Matrix([[1, 0], [0, 1], [1, 1]])
    q = sp.Matrix([1, 0])
    keys = sp.Matrix([[0, 0], [sp.sqrt(2)*sp.log(2), 0],
                      [sp.sqrt(2)*sp.log(3), 0]])
    queries = sp.Matrix.vstack(q.T, q.T, q.T)
    return scores, values, q, keys, queries


def _data(scores, values):
    scores = tuple(sp.sympify(s) for s in scores)
    values = sp.Matrix(values)
    if values.rows != len(scores) or values.cols < 1:
        raise ValueError("Values must have one row per score and positive width.")
    if any(x.is_real is not True or x.is_finite is not True
           for x in (*scores, *values)):
        raise ValueError("Scores and values require known finite real coordinates.")
    return scores, values


def _ids(ids, size, distinct=False):
    ids = tuple(ids)
    if any(type(j) is not int or j < 0 or j >= size for j in ids):
        raise ValueError("Record indices must be in-range integers.")
    if distinct and len(set(ids)) != len(ids):
        raise ValueError("Allowed sets and partitions cannot duplicate records.")
    return ids


@dataclass(frozen=True)
class Summary:
    maximum: sp.Expr
    denominator: sp.Expr
    numerator: sp.ImmutableMatrix


# None is a tagged empty identity, never an infinity-valued nonempty summary.
def summarize(ids, scores, values):
    scores, values = _data(scores, values)
    ids = _ids(ids, len(scores))
    if not ids:
        return None
    maximum = sp.Max(*(scores[j] for j in ids))
    weights = [sp.simplify(sp.exp(scores[j]-maximum)) for j in ids]
    numerator = sum((w*values[j, :].T for j, w in zip(ids, weights)),
                    sp.zeros(values.cols, 1))
    return Summary(maximum, sp.simplify(sum(weights)),
                   sp.ImmutableMatrix(numerator.applyfunc(sp.simplify)))


def merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a.numerator.shape != b.numerator.shape:
        raise ValueError("Summary value widths must match.")
    maximum = sp.Max(a.maximum, b.maximum)
    alpha, beta = sp.exp(a.maximum-maximum), sp.exp(b.maximum-maximum)
    return Summary(maximum, sp.simplify(alpha*a.denominator+beta*b.denominator),
                   sp.ImmutableMatrix((alpha*a.numerator+beta*b.numerator)
                                      .applyfunc(sp.simplify)))


def finalize(state):
    if state is None:
        raise ValueError("Attention over an empty allowed set is undefined.")
    return sp.Matrix(state.numerator/state.denominator).applyfunc(sp.simplify)


def reference_row(ids, scores, values):
    """Independent direct normalized-exponential reference, not a merge."""
    scores, values = _data(scores, values)
    ids = _ids(ids, len(scores), distinct=True)
    if not ids:
        raise ValueError("Attention needs at least one allowed record.")
    weights = [sp.exp(scores[j]) for j in ids]
    total = sum(weights)
    attention_coefficients = sp.zeros(len(scores), 1)
    for j, w in zip(ids, weights):
        attention_coefficients[j] = sp.simplify(w/total)
    return attention_coefficients, (values.T*attention_coefficients).applyfunc(sp.simplify)


def tiled_attention(queries, keys, values, allowed, query_tile, key_tile):
    """Typed tile traversal; returns output and a multiplicity coverage ledger."""
    if any(type(t) is not int or t < 1 for t in (query_tile, key_tile)):
        raise ValueError("Tile sizes must be positive integers.")
    queries, keys, values = map(sp.Matrix, (queries, keys, values))
    if (queries.rows < 1 or keys.rows < 1 or values.cols < 1 or
            queries.cols < 1 or queries.cols != keys.cols or
            keys.rows != values.rows or len(allowed) != queries.rows):
        raise ValueError("Incompatible attention shapes or mask row count.")
    if any(x.is_real is not True or x.is_finite is not True
           for matrix in (queries, keys, values) for x in matrix):
        raise ValueError("Attention inputs require known finite real coordinates.")
    supports = [_ids(row, keys.rows, distinct=True) for row in allowed]
    if any(not row for row in supports):
        raise ValueError("Every final row requires a nonempty allowed set.")
    output, coverage = sp.zeros(queries.rows, values.cols), Counter()
    scale = sp.sqrt(queries.cols)
    for row_start in range(0, queries.rows, query_tile):
        row_end = min(row_start+query_tile, queries.rows)
        active_queries = queries[row_start:row_end, :]
        states = {i: None for i in range(row_start, row_end)}
        for col_start in range(0, keys.rows, key_tile):
            col_end = min(col_start+key_tile, keys.rows)
            block_scores = active_queries * keys[col_start:col_end, :].T / scale
            block_values = values[col_start:col_end, :]
            for i in states:
                ids = tuple(j for j in range(col_start, col_end) if j in supports[i])
                coverage.update((i, j) for j in ids)
                local_ids = (j - col_start for j in ids)
                block = summarize(local_ids, block_scores.row(i - row_start), block_values)
                states[i] = merge(states[i], block)
        for i, state in states.items():
            output[i, :] = finalize(state).T
    return output, coverage


def distributed_row(shards, scores, values):
    """Independent check using a common maximum followed by aligned sums."""
    scores, values = _data(scores, values)
    shards = [_ids(ids, len(scores), distinct=True) for ids in shards]
    flat = [j for shard in shards for j in shard]
    if len(set(flat)) != len(flat):
        raise ValueError("Disjoint shard coverage is required.")
    if not flat:
        raise ValueError("All ranks are empty.")
    if set(flat) != set(range(len(scores))):
        raise ValueError("These shards must cover the declared full row.")
    local = [summarize(ids, scores, values) for ids in shards]
    maximum = sp.Max(*(s.maximum for s in local if s is not None))
    denominator, numerator = sp.Integer(0), sp.zeros(values.cols, 1)
    for state in local:
        if state is None:
            continue  # Zero contribution; never exponentiate an empty sentinel.
        scale = sp.exp(state.maximum-maximum)
        denominator += scale*state.denominator
        numerator += scale*state.numerator
    return (numerator/denominator).applyfunc(sp.simplify)


def reconstructed_probabilities(scores, log_normalizer):
    return sp.Matrix([sp.exp(s-log_normalizer) for s in scores]).applyfunc(sp.simplify)


def replayed_gradients(scores, values, g):
    scores, values = _data(scores, values)
    state = summarize(range(len(scores)), scores, values)
    o = finalize(state)
    attention_coefficients = reconstructed_probabilities(scores, state.maximum+sp.log(state.denominator))
    g = sp.Matrix(g)
    if g.shape != o.shape:
        raise ValueError("Upstream gradient width must match output.")
    ds = sp.Matrix([attention_coefficients[j]*(g.dot(values[j, :].T)-g.dot(o))
                    for j in range(len(scores))]).applyfunc(sp.simplify)
    return ds, (attention_coefficients*g.T).applyfunc(sp.simplify)


def tile_bytes(br, bc, dk=2, dv=2, io_bytes=2, acc_bytes=4):
    return io_bytes*(br*dk+bc*(dk+dv))+acc_bytes*(br*bc+br*dv+2*br)


def homework_arithmetic():
    """Part D: round operands and operations inside the selected mpmath context."""
    ordinary = ((1, 0), (0, 1), (1, 1))
    cases = [
        ('ordinary', (1, 2, 3), ordinary, 0),
        ('common shift', (1, 2, 3), ordinary, 1000),
        ('value scale', (1, 2, 3), tuple(tuple(10**6*x for x in row) for row in ordinary), 0),
        ('order-sensitive cancellation', (1, 1, 1),
         ((10**12, 1), (-10**12, 1), (1, -2)), 0),
    ]
    reports = []
    for label, exact_weights, values, shift in cases:
        # Exact direct row (known exp(log w) = w); no numerical reduction reused.
        with mp.workdps(70):
            high = tuple(mp.mpf(sum(w*row[a] for w, row in zip(exact_weights, values)))
                         / sum(exact_weights) for a in range(2))
        orders = [(0, 1, 2), (0, 2, 1)] if label == 'order-sensitive cancellation' else [(0, 1, 2)]
        for digits, order in product((8, 12), orders):
            with mp.workdps(digits):
                bits = mp.mp.prec
                scores = tuple(mp.log(mp.mpf(w)) + mp.mpf(shift) for w in exact_weights)
                numerical_values = tuple(tuple(map(mp.mpf, row)) for row in values)
                maximum = max(scores)
                weights = tuple(mp.exp(score-maximum) for score in scores)
                mass = mp.mpf(0)
                numerator = [mp.mpf(0), mp.mpf(0)]
                for i in order:
                    mass += weights[i]
                    for a in range(2):
                        numerator[a] += weights[i] * numerical_values[i][a]
                low = tuple(value / mass for value in numerator)
            accepted = normwise_agreement(low, high, '1e-7', '1e-5')
            with mp.workdps(70):
                finite = all(mp.isfinite(x) for x in (*low, *high))
                error = max(abs(x-y) for x, y in zip(low, high))
                reports.append(dict(case=label, backend='mpmath mpf',
                                    arithmetic='round to nearest at working binary precision',
                                    working_digits=digits, working_bits=bits, reference_digits=70,
                                    accumulation_order=[i+1 for i in order],
                                    output=[mp.nstr(x, digits) for x in low],
                                    reference=[mp.nstr(x, 70) for x in high],
                                    atol='1e-7', rtol='1e-5', finite=finite,
                                    max_absolute_error=mp.nstr(error, 12), accepted=accepted))
    return reports


def normwise_agreement(actual, reference, atol, rtol):
    """Finite max-entry norm rule, not elementwise allclose."""
    with mp.workdps(70):
        def number(x):
            if isinstance(x, sp.Basic):
                if x.is_real is not True or x.is_finite is not True:
                    return mp.nan
                return mp.mpf(str(sp.N(x, 70)))
            return mp.mpf(x)
        actual, reference = list(map(number, actual)), list(map(number, reference))
        atol, rtol = mp.mpf(atol), mp.mpf(rtol)
        if atol < 0 or rtol < 0 or not mp.isfinite(atol) or not mp.isfinite(rtol):
            raise ValueError("Tolerances must be finite and nonnegative.")
        if len(actual) != len(reference) or not actual:
            return False
        if not all(mp.isfinite(x) for x in actual+reference):
            return False
        error = max(abs(a-b) for a, b in zip(actual, reference))
        return error <= atol+rtol*max(abs(r) for r in reference)


def run_checks():
    exact, failures = [], []

    def equal(a, b, label):
        if isinstance(a, sp.MatrixBase) or isinstance(b, sp.MatrixBase):
            a, b = sp.Matrix(a), sp.Matrix(b)
            ok = a.shape == b.shape and all(sp.simplify(x) == 0 for x in a-b)
        else:
            ok = sp.simplify(a-b) == 0
        if not ok:
            raise AssertionError(label)
        exact.append(label)

    def rejects(call, label):
        try:
            call()
        except ValueError:
            exact.append(label)
        else:
            raise AssertionError(label)

    def same(a, b, label):
        equal(a.maximum, b.maximum, label + ': maximum')
        equal(a.denominator, b.denominator, label + ': denominator')
        equal(a.numerator, b.numerator, label + ': numerator')

    # A. Direct and online attention, then analytical backward.
    s, v, q, keys, queries = fixture()
    g = sp.Matrix([1, 2])
    expected_output = sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 6)])
    equal(keys*q/sp.sqrt(2), sp.Matrix(s), 'A: projected scores')
    probabilities, output = reference_row(range(3), s, v)
    equal(probabilities, sp.Matrix([sp.Rational(1, 6), sp.Rational(1, 3), sp.Rational(1, 2)]),
          'A: direct probabilities')
    equal(output, expected_output, 'A: direct output')
    rejects(lambda: reference_row((), s, v), 'A: all-empty direct rejection')

    trace, state = [], None
    for j in range(3):
        state = merge(state, summarize((j,), s, v))
        same(state, summarize(range(j + 1), s, v), f'A: online prefix {j + 1}')
        trace.append({'maximum': str(state.maximum), 'denominator': str(state.denominator),
                      'numerator': [str(x) for x in state.numerator]})
    full = state
    equal(full.maximum, sp.log(3), 'A: final maximum')
    equal(full.denominator, 2, 'A: final shifted mass')
    equal(full.numerator, sp.Matrix([sp.Rational(4, 3), sp.Rational(5, 3)]),
          'A: final shifted numerator')
    equal(sp.exp(full.maximum)*full.denominator, 6, 'A: raw mass')
    equal(sp.exp(full.maximum)*full.numerator, sp.Matrix([4, 5]), 'A: raw numerator')
    equal(finalize(full), expected_output, 'A: online output')

    first = summarize((2,), s, v)
    smaller = merge(first, summarize((0,), s, v))
    equal(smaller.maximum, first.maximum, 'A: smaller score leaves the maximum')
    equal(smaller.denominator, sp.Rational(4, 3), 'A: smaller score updates the mass')
    equal(smaller.numerator, sp.Matrix([sp.Rational(4, 3), 1]),
          'A: smaller score updates the numerator')

    score_symbols = sp.symbols('s0:3', real=True)
    value_symbols = sp.Matrix(3, 2, sp.symbols('v0:6', real=True))
    generic_probabilities, generic_output = reference_row(range(3), score_symbols, value_symbols)
    loss = g.dot(generic_output)
    symbolic_ds = sp.Matrix([sp.diff(loss, score) for score in score_symbols])
    symbolic_dv = value_symbols.applyfunc(lambda coordinate: sp.diff(loss, coordinate))
    analytic_ds = sp.Matrix([
        generic_probabilities[j]*g.dot(value_symbols.row(j).T - generic_output)
        for j in range(3)
    ])
    equal(symbolic_ds, analytic_ds, 'A: generic score derivative versus quotient formula')
    equal(symbolic_dv, generic_probabilities*g.T, 'A: generic value derivative')
    value_point = dict(zip(value_symbols, v))
    for label, scores in [('worked row', s), ('tied scores', (sp.Integer(0),)*3)]:
        point = value_point | dict(zip(score_symbols, scores))
        ds, dv = replayed_gradients(scores, v, g)
        equal(ds, symbolic_ds.subs(point), f'A: reconstructed score gradient, {label}')
        equal(dv, symbolic_dv.subs(point), f'A: reconstructed value gradient, {label}')
        equal(sum(ds), 0, f'A: zero score-gradient sum, {label}')

    ds, dv = replayed_gradients(s, v, g)
    equal(ds, sp.Matrix([-sp.Rational(2, 9), -sp.Rational(1, 9), sp.Rational(1, 3)]),
          'A: score-gradient witness')
    equal(dv, probabilities*g.T, 'A: value-gradient witness')
    equal(sum(reconstructed_probabilities((*s[:2], sp.log(6)), sp.log(6))),
          sp.Rational(3, 2), 'A: stale normalizer mass')
    failures.append({'fault': 'stale log-normalizer', 'witness': 'mass=3/2'})
    frozen_ds = probabilities.multiply_elementwise(v*g)
    equal(frozen_ds-ds, probabilities*g.dot(output), 'A: detached normalizer loses coupling')

    # Exercise 19.4: only the second causal query has nonzero upstream gradient.
    query_symbols = sp.Matrix(sp.symbols('q0:2', real=True))
    key_symbols = sp.Matrix(3, 2, sp.symbols('k0:6', real=True))
    scale = sp.sqrt(2)
    _, masked_output = reference_row((0, 1), key_symbols*query_symbols/scale, value_symbols)
    masked_loss = g.dot(masked_output)
    point = value_point | dict(zip(query_symbols, q)) | dict(zip(key_symbols, keys))
    masked_ds, masked_dv = replayed_gradients(s[:2], v[:2, :], g)
    masked_ds = masked_ds.col_join(sp.zeros(1, 1))
    masked_dv = masked_dv.col_join(sp.zeros(1, v.cols))
    dq = keys.T*masked_ds/scale
    dk = masked_ds*q.T/scale
    for name, symbols, analytical in [('Q', query_symbols, dq), ('K', key_symbols, dk),
                                       ('V', value_symbols, masked_dv)]:
        direct = symbols.applyfunc(lambda x: sp.diff(masked_loss, x).subs(point))
        equal(analytical, direct, f'A: masked {name} chain rule versus direct differentiation')

    # B. Block merging and causal tile coverage.
    for order in permutations(range(3)):
        a, b, c = [summarize((j,), s, v) for j in order]
        same(merge(merge(a, b), c), full, f'B: left tree {order}')
        same(merge(a, merge(b, c)), full, f'B: right tree {order}')
        for cuts in range(4):
            chunks, current = [], []
            for pos, j in enumerate(order):
                current.append(j)
                if pos == 2 or cuts & (1 << pos):
                    chunks.append(tuple(current))
                    current = []
            state = None
            for chunk in chunks:
                state = merge(state, summarize(chunk, s, v))
            same(state, full, f'B: chunking {order}/{cuts}')

    same(merge(None, full), full, 'B: left empty identity')
    same(merge(full, None), full, 'B: right empty identity')
    if merge(None, None) is not None:
        raise AssertionError('B: two empty summaries')
    exact.append('B: two empty identities')
    rejects(lambda: finalize(None), 'B: empty finalization rejects')

    # Verify scale alignment in both sums without choosing concrete scores.
    ma, mb, common = sp.symbols('ma mb common', real=True)
    a1, a2, b1 = sp.symbols('a1 a2 b1', real=True)
    va, vb, vc = sp.symbols('va vb vc', real=True)
    ea, eb = sp.exp(a1-ma)+sp.exp(a2-ma), sp.exp(b1-mb)
    equal(sp.exp(ma-common)*ea + sp.exp(mb-common)*eb,
          sum(sp.exp(t-common) for t in (a1, a2, b1)), 'B: symbolic merge denominator')
    equal(sp.exp(ma-common)*(sp.exp(a1-ma)*va + sp.exp(a2-ma)*vb) + sp.exp(mb-common)*eb*vc,
          sum(sp.exp(t-common)*w for t, w in zip((a1, a2, b1), (va, vb, vc))),
          'B: symbolic merge numerator coordinate')

    unequal = ((0,), (1, 2))
    balanced = ((0, 1), (2,))
    for partition in (unequal, balanced, ((), (0, 1, 2))):
        equal(distributed_row(partition, s, v), expected_output,
              f'B: independent common-scale calculation {partition}')
    for partition, expected in ((unequal, sp.Matrix([sp.Rational(4, 5), sp.Rational(1, 2)])),
                                (balanced, expected_output)):
        a, b = (summarize(ids, s, v) for ids in partition)
        average = (finalize(a) + finalize(b))/2
        equal(average, expected, f'B: unweighted average {partition}')
        if partition == unequal:
            failures.append({'fault': 'unweighted block averaging', 'witness': [str(x) for x in average]})
    a, b = (summarize(ids, s, v) for ids in balanced)
    raw = (a.numerator+b.numerator)/(a.denominator+b.denominator)
    equal(raw, sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)]), 'B: unaligned shifted sums')
    failures.append({'fault': 'unaligned shifted sums', 'witness': [str(x) for x in raw]})

    allowed = [tuple(range(i + 1)) for i in range(3)]
    expected_coverage = Counter((i, j) for i, row in enumerate(allowed) for j in row)
    expected_causal = sp.Matrix.vstack(*(reference_row(row, s, v)[1].T for row in allowed))
    changed_values = v.copy()
    changed_values[2, :] = sp.Matrix([[7, -4]])
    expected_changed = sp.Matrix.vstack(
        *(reference_row(row, s, changed_values)[1].T for row in allowed)
    )
    equal(expected_changed.row(2), sp.Matrix([[sp.Rational(11, 3), -sp.Rational(5, 3)]]),
          'B: changed third row')
    for br, bc in product((1, 2, 4), repeat=2):
        out, coverage = tiled_attention(queries, keys, v, allowed, br, bc)
        equal(out, expected_causal, f'B: tiled versus direct {br}/{bc}')
        changed, changed_coverage = tiled_attention(queries, keys, changed_values, allowed, br, bc)
        equal(changed, expected_changed, f'B: future-value change versus direct {br}/{bc}')
        equal(changed[:2, :], out[:2, :], f'B: earlier causal rows {br}/{bc}')
        if coverage != expected_coverage or changed_coverage != expected_coverage:
            raise AssertionError('B: exact-once pair coverage')
        exact.append(f'B: six allowed pairs once {br}/{bc}')

    reversed_keys, reversed_values = keys[::-1, :], v[::-1, :]
    original_support = [tuple(2 - j for j in row) for row in allowed]
    preserved, _ = tiled_attention(queries, reversed_keys, reversed_values, original_support, 2, 2)
    relabeled, _ = tiled_attention(queries, reversed_keys, reversed_values, allowed, 2, 2)
    equal(preserved, expected_causal, 'B: reverse records with original support')
    equal(relabeled.row(0), v.row(2), 'B: causal mask rebuilt from physical order')

    # C. Contraction work, tile feasibility and backing-memory traffic.
    n, dk, dv = 8, 2, 2
    io_bytes, fast_memory = 2, 96
    issued_pairs = sum((block_row + 1)*4 for block_row in range(n//2))
    workloads = (
        ('full_logical', n*n, 64, 512),
        ('causal_logical', n*(n + 1)//2, 36, 288),
        ('causal_issued', issued_pairs, 40, 320),
    )
    pair_counts = {}
    contraction_flops = {}
    for name, pairs, expected_pairs, expected_work in workloads:
        equal(pairs, expected_pairs, f'C: {name} pairs')
        pair_counts[name] = pairs
        contraction_flops[name] = 2*pairs*(dk + dv)
        equal(contraction_flops[name], expected_work, f'C: {name} contraction FLOPs')
    equal(2*io_bytes*n*n, 256, 'C: separate live arrays')
    equal(4*io_bytes*n*n, 512, 'C: isolated materialized traffic')

    tiles = {}
    for br, expected_footprint, expected_traffic in ((2, 72, 320), (4, 128, 192)):
        footprint = tile_bytes(br, 2, dk, dv, io_bytes, acc_bytes=4)
        sweeps = (n + br - 1)//br
        traffic = io_bytes*(n*dk + n*dv + sweeps*n*(dk + dv))
        equal(footprint, expected_footprint, f'C: tile {br}x2 footprint')
        equal(traffic, expected_traffic, f'C: tile {br}x2 no-spill traffic')
        tiles[f'{br}x2'] = {'fast_memory_bytes': footprint, 'fits_96_bytes': footprint <= fast_memory,
                           'no_spill_traffic_bytes': traffic}

    peak_flops, bandwidth = sp.Integer(120)*10**12, sp.Integer(3)*10**12
    work = sp.Integer(12)*10**9
    equal(peak_flops/bandwidth, 40, 'C: roofline ridge')
    bounds = []
    for traffic, intensity, seconds in [(sp.Integer(12)*10**8, 10, sp.Rational(1, 2500)),
                                        (sp.Integer(3)*10**8, 40, sp.Rational(1, 10000))]:
        equal(work/traffic, intensity, f'C: intensity {intensity}')
        bound = max(work/peak_flops, traffic/bandwidth)
        equal(bound, seconds, f'C: service bound {seconds} seconds')
        bounds.append({'traffic_bytes': int(traffic), 'flops_per_byte': intensity,
                       'seconds': str(bound)})

    # D. Exact invariances and the declared finite-precision comparisons.
    shift, scale = sp.symbols('shift scale', real=True)
    _, shifted = reference_row(range(3), [score+shift for score in score_symbols], value_symbols)
    _, scaled = reference_row(range(3), score_symbols, scale*value_symbols)
    equal(shifted, generic_output, 'D: generic common-score shift')
    equal(scaled, scale*generic_output, 'D: generic value scale')

    arithmetic = homework_arithmetic()
    for row in arithmetic:
        loses_small_term = (
            row['case'] == 'order-sensitive cancellation'
            and row['working_digits'] == 8
            and row['accumulation_order'] == [1, 3, 2]
        )
        expected_acceptance = not loses_small_term
        if not row['finite'] or row['accepted'] != expected_acceptance:
            raise AssertionError(f'D: unexpected numerical comparison {row}')
    if normwise_agreement([mp.inf], [0], '1e-7', '1e-5'):
        raise AssertionError('D: nonfinite comparison')
    fa, fb = float(2**54), -float(2**54)
    if sys.float_info.mant_dig != 53 or (fa+fb)+1.0 != 1.0 or fa+(fb+1.0) != 0.0:
        raise AssertionError('D: declared binary64 witness unavailable')

    return {
        'python': sys.version.split()[0], 'sympy': sp.__version__, 'mpmath': mp.__version__,
        'exact_checks_passed': len(exact), 'exact_checks': exact,
        'finite_sweep': {'paired_permutations': 6, 'binary_trees': 12, 'chunkings': 24,
                         'tile_shapes': 9, 'allowed_pairs_per_shape': 6,
                         'causality_perturbation_shapes': 9},
        'prefix_trace': trace, 'deliberate_fault_witnesses': failures,
        'C_resources': {'pair_counts': pair_counts, 'contraction_flops': contraction_flops,
                        'tiles': tiles, 'resource_bounds': bounds},
        'D_working_precision': arithmetic, 'D_binary64_sums': [1.0, 0.0],
        'scope': 'Exact calculations, finite examples and modeled resource bounds.',
    }


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2))
