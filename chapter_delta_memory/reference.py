"""Instructor's exact laboratory; no trained-model or timing claims.

The sequential, affine-prefix, and triangular implementations are independent
organizations of the equations. Matrices use value-by-key orientation.
"""
from itertools import product
import sys
import sympy as sp
from student_setup import exact_equal, gated_chunk_fixture, keys, values, queries, betas, gates, S0

R = sp.Rational


def check_shapes(S, k, v, q, D):
    dv, dk = S.shape
    for name, matrix, shape in (
        ("key", k, (dk, 1)), ("value", v, (dv, 1)),
        ("query", q, (dk, 1)), ("gate", D, (dk, dk)),
    ):
        if matrix.shape != shape:
            raise ValueError(f"{name} shape {matrix.shape}; expected {shape}")
    if any(D[i, j] != 0 for i in range(dk) for j in range(dk) if i != j):
        raise ValueError("retention gate must be diagonal")


def step(S, k, v, q, beta=1, D=None, additive=False):
    """Return retained state, prediction, error, new state, post-write read.

    In additive mode beta scales the whole value write (beta=1 is the baseline).
    Scalars may be symbolic; the mathematical stability bounds are separate.
    """
    if D is None:
        D = sp.eye(S.cols)
    check_shapes(S, k, v, q, D)
    retained = S * D
    prediction = retained * k
    error = v - prediction
    updated = retained + beta * (v if additive else error) * k.T
    return retained, prediction, error, updated, updated * q


def step_pullback(S, k, v, q, beta, D, output_bar, later_state_bar):
    """Return adjoints of S, diag(D), k, v, q and beta in that order."""
    retained, _, error, updated, _ = step(S, k, v, q, beta, D)
    total_state_bar = later_state_bar + output_bar * q.T
    query_bar = updated.T * output_bar
    value_bar = beta * total_state_bar * k
    strength_bar = error.dot(total_state_bar * k)
    key_bar = beta * total_state_bar.T * error - retained.T * value_bar
    retained_bar = total_state_bar - value_bar * k.T
    state_bar = retained_bar * D.T
    gate_bar = sp.Matrix((S.T * retained_bar).diagonal()).T
    return state_bar, gate_bar, key_bar, value_bar, query_bar, strength_bar


def sequential(initial, ks, vs, qs, bs, ds, additive=False):
    lengths = {len(x) for x in (ks, vs, qs, bs, ds)}
    if len(lengths) != 1:
        raise ValueError("coefficient streams must have equal lengths")
    state = initial
    states, outputs = [], []
    for k, v, q, beta, D in zip(ks, vs, qs, bs, ds):
        *_, state, output = step(state, k, v, q, beta, D, additive)
        states.append(state)
        outputs.append(output)
    return states, outputs


def compose(earlier, later):
    """Chronological affine composition: apply earlier, then later."""
    A1, B1 = earlier
    A2, B2 = later
    return A1 * A2, B1 * A2 + B2


def affine_chunks(initial, ks, vs, qs, bs, ds, partition):
    if len({len(x) for x in (ks, vs, qs, bs, ds)}) != 1:
        raise ValueError("coefficient streams must have equal lengths")
    if any(not isinstance(c, int) or c <= 0 for c in partition):
        raise ValueError("chunk lengths must be positive integers")
    if sum(partition) != len(ks):
        raise ValueError("partition must cover every event exactly once")
    state, offset = initial, 0
    states, outputs = [], []
    for length in partition:
        incoming = state
        prefix = (sp.eye(initial.cols), sp.zeros(*initial.shape))
        for t in range(offset, offset + length):
            k, v, q, beta, D = ks[t], vs[t], qs[t], bs[t], ds[t]
            check_shapes(incoming, k, v, q, D)
            A = D * (sp.eye(initial.cols) - beta * k * k.T)
            B = beta * v * k.T
            prefix = compose(prefix, (A, B))
            state = incoming * prefix[0] + prefix[1]
            states.append(state)
            outputs.append(state * q)
        offset += length
    return states, outputs


def triangular_chunk(incoming, ks, vs, qs, bs, ds=None):
    """Solve for gated writes without calling step or compose."""
    C = len(ks)
    if ds is None:
        ds = [sp.eye(incoming.cols)] * C
    if C == 0 or len({C, len(vs), len(qs), len(bs), len(ds)}) != 1:
        raise ValueError("one nonempty chunk with equal stream lengths required")
    G = sp.zeros(C)
    read_coefficients = sp.zeros(C)
    B = sp.zeros(C, incoming.rows)
    incoming_reads = sp.zeros(C, incoming.rows)
    retained_incoming = incoming
    write_keys = []
    for i, (k, v, q, beta, D) in enumerate(zip(ks, vs, qs, bs, ds)):
        check_shapes(incoming, k, v, q, D)
        retained_incoming = retained_incoming * D
        write_keys = [D * earlier for earlier in write_keys]
        for j, earlier in enumerate(write_keys):
            G[i, j] = beta * earlier.dot(k)
            read_coefficients[i, j] = earlier.dot(q)
        read_coefficients[i, i] = k.dot(q)
        write_keys.append(k)
        B[i, :] = (beta * (v - retained_incoming * k)).T
        incoming_reads[i, :] = (retained_incoming * q).T

    U = (sp.eye(C) + G).lower_triangular_solve(B)
    outgoing = retained_incoming + U.T * sp.Matrix.vstack(*(key.T for key in write_keys))
    O = incoming_reads + read_coefficients * U
    return outgoing, [O.row(i).T for i in range(C)], G, U


def triangular_chunks(initial, ks, vs, qs, bs, partition, ds=None):
    if ds is not None and len(ds) != len(ks):
        raise ValueError("coefficient streams must have equal lengths")
    if any(not isinstance(c, int) or c <= 0 for c in partition):
        raise ValueError("chunk lengths must be positive integers")
    if sum(partition) != len(ks):
        raise ValueError("partition must cover every event exactly once")
    state, offset, outputs = initial, 0, []
    for C in partition:
        sl = slice(offset, offset + C)
        local_gates = None if ds is None else ds[sl]
        state, local, _, _ = triangular_chunk(state, ks[sl], vs[sl], qs[sl], bs[sl], local_gates)
        outputs.extend(local)
        offset += C
    return state, outputs


def five_event_fixture():
    """Nonzero 3-by-2 initial state; overlap, zero key/gate/beta, short tails."""
    initial = sp.Matrix([[1, R(1, 2)], [-1, 2], [R(2, 3), -R(1, 3)]])
    a, b, overlap, zero = (sp.Matrix(x) for x in ([1, 0], [0, 1], [R(3, 5), R(4, 5)], [0, 0]))
    ks = [a, overlap, b, zero, a]
    vs = [sp.Matrix(x) for x in ([1, 2, 3], [0, 1, -1], [2, 0, 1], [1, 1, 1], [-1, 2, 0])]
    qs = [b, a, overlap, b, a]
    bs = [1, R(1, 2), 0, 1, R(3, 4)]
    ds = [sp.eye(2), sp.diag(R(1, 2), R(3, 4)), sp.diag(0, 1), sp.eye(2), sp.diag(1, R(1, 2))]
    return initial, ks, vs, qs, bs, ds


def all_partitions(n):
    """All 2**(n-1) contiguous nonempty partitions of n events."""
    for cuts in product((False, True), repeat=n - 1):
        partition, start = [], 0
        for end, cut in enumerate(cuts, 1):
            if cut:
                partition.append(end - start)
                start = end
        yield partition + [n - start]


def run_reference():
    checks = 0

    def equal(actual, expected):
        nonlocal checks
        if not exact_equal(actual, expected):
            raise AssertionError(f"Exact residual: {actual - expected}")
        checks += 1

    # A/C: independently asserted states, paired controls, typed rectangular case.
    I, a, b = sp.eye(2), keys[0], keys[1]
    ones = [sp.Integer(1)] * 3
    identities = [I] * 3
    additive, add_reads = sequential(S0, keys, values, queries, ones, identities, True)
    delta, delta_reads = sequential(S0, keys, values, queries, ones, identities)
    gated, gated_reads = sequential(S0, keys, values, queries, betas, gates)
    equal(additive[-1], sp.Matrix([[1, 0], [1, 1]]))
    equal(delta[-1], sp.Matrix([[0, 0], [1, 1]]))
    equal(gated[-1], sp.Matrix([[R(1, 4), 0], [R(1, 2), R(3, 4)]]))
    equal(gated_reads[-1], sp.Matrix([R(1, 4), R(1, 2)]))
    equal(step(I, a, b, a, R(1, 2))[3], sp.Matrix([[R(1, 2), 0], [R(1, 2), 1]]))
    equal(step(I, a, b, a, R(1, 2), I / 2)[3], sp.Matrix([[R(1, 4), 0], [R(1, 2), R(1, 2)]]))
    equal(step(I, a, b, a, R(1, 2), gates[2], True)[3], sp.Matrix([[R(1, 2), 0], [R(1, 2), R(3, 4)]]))
    equal(step(sp.zeros(3, 2), a, sp.Matrix([1, 2, 3]), a)[4], sp.Matrix([1, 2, 3]))
    k = sp.Matrix([R(3, 5), R(4, 5)])
    P, D = I - k * k.T, gates[2]
    overlap = step(I, k, b, a)[3]
    equal(overlap, sp.Matrix([[R(16, 25), -R(12, 25)], [R(3, 25), R(29, 25)]]))
    equal(overlap * k, b)
    commutator = D * P - P * D
    equal(commutator, sp.Matrix([[0, R(3, 25)], [-R(3, 25), 0]]))
    correct = step(I, k, b, a, 1, D)[3]
    swapped = I * P * D + b * k.T
    if exact_equal(correct, swapped):
        raise AssertionError("swapped gate order escaped detection")
    checks += 1
    equal(step(I, sp.zeros(2, 1), b, a, 1, D)[3], I * D)
    equal(step(I, k, b, a, 0, D)[3], I * D)
    equal(step(I, a, b, b, 0, sp.diag(0, 1))[3], sp.diag(0, 1))
    equal(step(I, a, b, a, 1, sp.zeros(2))[3], b * a.T)
    # B: derive the matrix gradient from six independent entries.
    S = sp.Matrix(3, 2, sp.symbols("s0:6", real=True))
    key = sp.Matrix(sp.symbols("k0:2", real=True))
    value = sp.Matrix(sp.symbols("v0:3", real=True))
    query = sp.Matrix(sp.symbols("q0:2", real=True))
    beta = sp.symbols("beta", real=True)
    residual = value - S * key
    loss = (residual.T * residual)[0] / 2
    gradient = S.applyfunc(lambda entry: sp.diff(loss, entry))
    equal(gradient, (S * key - value) * key.T)
    plus = S + beta * residual * key.T
    equal(value - plus * key, (1 - beta * (key.T * key)[0]) * residual)
    equal(plus * query - S * query, beta * residual * (key.T * query)[0])
    nonunit = step(S0, 2 * a, a, a)[3]
    equal(a - nonunit * (2 * a), -3 * a)
    equal(step(S0, 2 * a, a, a, R(1, 4))[3] * (2 * a), a)
    # D: characteristic polynomial and homogeneous/driven distinction.
    z = sp.symbols("z")
    symbolic_P = I - beta * k * k.T
    if sp.simplify(symbolic_P.charpoly(z).as_expr() - (z - 1) * (z - 1 + beta)) != 0:
        raise AssertionError("incorrect characteristic polynomial")
    checks += 1
    driven = step(S0, a, a, a)[3]
    if driven.norm() <= S0.norm():
        raise AssertionError("new write must increase this zero state's norm")
    checks += 1
    # E: exact same-input affine prefixes, including all five-event partitions.
    for partition in ([3], [2, 1]):
        states, outputs = affine_chunks(S0, keys, values, queries, betas, gates, partition)
        for actual, expected in zip(states + outputs, gated + gated_reads):
            equal(actual, expected)
    initial, ks, vs, qs, bs, ds = five_event_fixture()
    states, outputs = sequential(initial, ks, vs, qs, bs, ds)
    for partition in all_partitions(5):
        chunk_states, chunk_outputs = affine_chunks(initial, ks, vs, qs, bs, ds, partition)
        for actual, expected in zip(chunk_states + chunk_outputs, states + outputs):
            equal(actual, expected)
    for incoming, case_keys, case_values, case_queries, case_betas in (
        (S0, keys, values, queries, ones),
        (I, keys, values, queries, betas),
        (initial, ks, vs, qs, bs),
    ):
        refs, reads = sequential(incoming, case_keys, case_values, case_queries, case_betas,
                                 [sp.eye(incoming.cols)] * len(case_keys))
        for partition in all_partitions(len(case_keys)):
            final, local = triangular_chunks(incoming, case_keys, case_values, case_queries,
                                             case_betas, partition)
            for actual, expected in zip([final] + local, [refs[-1]] + reads):
                equal(actual, expected)
    outgoing, all_reads, G, U = triangular_chunk(S0, keys, values, queries, ones)
    equal(G, sp.Matrix([[0, 0, 0], [0, 0, 0], [1, 0, 0]]))
    equal(U, sp.Matrix([[1, 0], [0, 1], [-1, 1]]))
    if exact_equal(outgoing * queries[0], all_reads[0]):
        raise AssertionError("final-state-for-all-queries bug escaped detection")
    checks += 1
    # E3: a gated triangular solve with a short tail and a nonzero boundary.
    initial, ks, vs, qs, bs, ds = gated_chunk_fixture()
    states, reads = sequential(initial, ks, vs, qs, bs, ds)
    for partition in all_partitions(3):
        final, local = triangular_chunks(initial, ks, vs, qs, bs, partition, ds)
        for actual, expected in zip([final] + local, [states[-1]] + reads):
            equal(actual, expected)
    _, reset_reads, _, _ = triangular_chunk(sp.zeros(2), ks[2:], vs[2:], qs[2:], bs[2:], ds[2:])
    if exact_equal(reset_reads[0], reads[-1]):
        raise AssertionError("resetting the chunk boundary escaped detection")
    checks += 1

    # E4: differentiate the scalar forward loss before substituting event 2.
    retention = sp.Matrix(sp.symbols("a0:2", real=True))
    upstream = sp.Matrix([1, 2])
    later = sp.Matrix([[1, 0], [1, 1]])
    symbolic_value = value[:2, :]
    local_S = S[:2, :]
    local_inputs = list(local_S) + list(retention) + list(key) + list(symbolic_value) + list(query) + [beta]
    _, _, _, local_out, local_read = step(local_S, key, symbolic_value, query, beta, sp.diag(*retention))
    loss = upstream.dot(local_read) + sp.trace(later.T * local_out)
    substitutions = dict(zip(local_inputs,
                             list(states[0]) + list(ds[1].diagonal()) + list(ks[1])
                             + list(vs[1]) + list(qs[1]) + [bs[1]]))
    adjoints = step_pullback(states[0], ks[1], vs[1], qs[1], bs[1], ds[1], upstream, later)
    flattened = [entry for matrix in adjoints[:-1] for entry in matrix] + [adjoints[-1]]
    for var, expected in zip(local_inputs, flattened):
        equal(sp.Matrix([sp.diff(loss, var).subs(substitutions)]), sp.Matrix([expected]))

    # Associativity on arbitrary symbolic matrices, not only the toy stream.
    As = [sp.Matrix(2, 2, sp.symbols(f"a{i}_0:4")) for i in range(3)]
    Bs = [sp.Matrix(3, 2, sp.symbols(f"b{i}_0:6")) for i in range(3)]
    f, g, h = list(zip(As, Bs))
    left, right = compose(compose(f, g), h), compose(f, compose(g, h))
    equal(left[0], right[0])
    equal(left[1], right[1])
    for invalid_k, invalid_v, invalid_q, invalid_D in (
        (sp.zeros(3, 1), b, a, I), (a, sp.zeros(3, 1), a, I),
        (a, b, sp.zeros(1, 2), I), (a, b, a, sp.eye(3)),
    ):
        try:
            step(I, invalid_k, invalid_v, invalid_q, 1, invalid_D)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("incompatible shape accepted")
    # F: element count is distinct from an allocated-byte or timing measurement.
    T = sp.symbols("T", positive=True)
    hybrid, full = 12 + 4 * T, 16 * T
    if hybrid.subs(T, 100) != 412 or full.subs(T, 100) != 1600:
        raise AssertionError("hybrid accounting mismatch")
    if sp.limit(hybrid / full, T, sp.oo) != R(1, 4):
        raise AssertionError("hybrid limiting ratio mismatch")
    checks += 3
    print("Python:", sys.version.split()[0], "SymPy:", sp.__version__)
    print(f"PASS: {checks} exact checks; all 16 five-event contiguous partitions.")
    print("Gated revised-A state:", gated[-1])
    print("Bug witnesses: swapped gate order and final-state prefix reads detected.")
    return checks


if __name__ == "__main__":
    run_reference()
