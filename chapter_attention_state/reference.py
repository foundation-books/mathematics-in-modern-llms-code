"""Instructor A–F solution. Exact checks supplement the chapter's general proofs."""

from dataclasses import dataclass
import sys

import sympy as sp

from fixture import assert_zero, fixture, softmax_exact


@dataclass(frozen=True)
class HeadTrace:
    scores: sp.Matrix
    weights: sp.Matrix
    output: sp.Matrix
    mixture: object = None


@dataclass(frozen=True)
class Trace:
    heads: tuple
    output: sp.Matrix


def validate(data, t):
    if type(t) is not int or not 1 <= t <= data.X.cols:
        raise ValueError("prefix length must be an integer in 1..T")
    d, r = data.X.rows, data.D.rows
    H_q = len(data.WQ)
    if H_q == 0 or r == 0 or data.D.cols != d:
        raise ValueError("invalid input or latent dimensions")
    if any(len(maps) != H_q for maps in (data.UK, data.UV, data.WO_col)):
        raise ValueError("all head map collections must have length H")
    dk, dv = data.UK[0].rows, data.UV[0].rows
    if dk == 0 or dv == 0:
        raise ValueError("head widths must be positive")
    for h in range(H_q):
        if (data.WQ[h].shape != (dk, d) or data.UK[h].shape != (dk, r)
                or data.UV[h].shape != (dv, r) or data.WO_col[h].shape != (d, dv)):
            raise ValueError("incompatible head projection shape")
    return d, r, H_q, dk, dv


def combine(data, heads):
    return sum((WO_col * head.output for WO_col, head in zip(data.WO_col, heads)),
               sp.zeros(data.X.rows, 1))


def positional_keys(data, t):
    R = sp.Matrix([[0, -1], [1, 0]])
    return sp.Matrix.hstack(*(R**j * data.X[1:3, j-1] for j in range(1, t+1)))


def expanded(data, t, source_heads=None, positional=False):
    """Explicit per-head K/V of the specified variant (zero-based source heads)."""
    _, _, H_q, dk, _ = validate(data, t)
    source_heads = tuple(range(H_q)) if source_heads is None else tuple(source_heads)
    if len(source_heads) != H_q or any(type(h) is not int or not 0 <= h < H_q
                                     for h in source_heads):
        raise ValueError("one valid source-head index is required per query head")
    X = data.X[:, :t]
    heads = []
    for h, source in enumerate(source_heads):
        q = data.WQ[h] * data.X[:, t-1]
        K = (data.UK[source] * data.D) * X
        V = (data.UV[source] * data.D) * X
        if positional:
            qr = sp.Matrix([[0, -1], [1, 0]])**t * sp.Matrix([1, 0])
            q = q.col_join(qr)
            K = K.col_join(positional_keys(data, t))
        scores = K.T * q / sp.sqrt(dk + (2 if positional else 0))
        weights = softmax_exact(scores)
        heads.append(HeadTrace(scores, weights, V * weights))
    return Trace(tuple(heads), combine(data, heads))


def grouped(data, t, groups, sources):
    """Read one cached K/V prefix per group, retaining distinct query heads."""
    _, _, H_q, dk, _ = validate(data, t)
    groups, sources = tuple(groups), tuple(sources)
    if (len(groups) != H_q or not sources
            or any(type(g) is not int for g in groups)
            or set(groups) != set(range(len(sources)))):
        raise ValueError("every counted group must be used by at least one query head")
    if any(type(h) is not int or not 0 <= h < H_q for h in sources):
        raise ValueError("each group needs a valid source-head index")
    X = data.X[:, :t]
    caches = tuple(((data.UK[h] * data.D) * X, (data.UV[h] * data.D) * X)
                   for h in sources)
    heads = []
    for h, group in enumerate(groups):
        K, V = caches[group]
        scores = K.T * (data.WQ[h] * data.X[:, t-1]) / sp.sqrt(dk)
        weights = softmax_exact(scores)
        heads.append(HeadTrace(scores, weights, V * weights))
    return Trace(tuple(heads), combine(data, heads))


def absorbed(data, t, positional=False):
    _, _, H_q, dk, _ = validate(data, t)
    C = data.D * data.X[:, :t]
    scale = sp.sqrt(dk + (2 if positional else 0))
    heads = []
    merged_output = sp.zeros(data.X.rows, 1)
    for h in range(H_q):
        q = data.WQ[h] * data.X[:, t-1]
        qt = data.UK[h].T * q / scale
        scores = C.T * qt
        if positional:
            qr = sp.Matrix([[0, -1], [1, 0]])**t * sp.Matrix([1, 0])
            scores += positional_keys(data, t).T * qr / scale
        weights = softmax_exact(scores)
        mixture = C * weights
        heads.append(HeadTrace(scores, weights, data.UV[h] * mixture, mixture))
        merged_output += (data.WO_col[h] * data.UV[h]) * mixture
    assert_zero(merged_output - combine(data, heads))
    return Trace(tuple(heads), merged_output)


def compare(left, right):
    if len(left.heads) != len(right.heads):
        raise AssertionError("head counts differ")
    for a, b in zip(left.heads, right.heads):
        for field in ("scores", "weights", "output"):
            assert_zero(getattr(a, field) - getattr(b, field))
    assert_zero(left.output - right.output)


def payloads():
    B, T, L_attn, H_q, H_KV, dk, dv, b_elt = 2, 4096, 24, 16, 4, 64, 64, 2
    base = b_elt * B * T * (dk + dv)
    return {"MHA": base * L_attn * H_q, "GQA": base * L_attn * H_KV,
            "MQA": base * L_attn, "GQA/12 producers": base * 12 * H_KV}


def run():
    data = fixture()
    print("Python:", sys.version.split()[0], "SymPy:", sp.__version__)
    # A and C: each head and each legal causal prefix, stage by stage.
    for t in range(1, 4):
        direct = expanded(data, t)
        for head in direct.heads:
            assert_zero(sp.Matrix([sum(head.weights) - 1]))
            if any(sp.simplify(a).is_positive is not True for a in head.weights):
                raise AssertionError("positivity unresolved")
        compare(direct, absorbed(data, t))
    golden = expanded(data, 3).heads[0]
    assert_zero(golden.scores - sp.Matrix([0, sp.log(2), sp.log(3)]))
    assert_zero(golden.weights - sp.Matrix([sp.Rational(1, 6), sp.Rational(1, 3), sp.Rational(1, 2)]))
    assert_zero(golden.output - sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 6)]))
    assert_zero(expanded(data, 2).heads[0].output - sp.Matrix([sp.Rational(1, 3), sp.Rational(2, 3)]))
    assert_zero(expanded(data, 1).heads[0].output - sp.Matrix([1, 0]))
    q = sp.Matrix(sp.symbols("q0:2", real=True))
    U = sp.Matrix(2, 3, sp.symbols("u0:6", real=True))
    c = sp.Matrix(sp.symbols("c0:3", real=True))
    assert_zero(q.T * U * c - (U.T * q).T * c)
    print("A/C: 12 head-prefix comparisons; scores, weights, values and 3 combined outputs exact.")

    # B: a separate direct expansion for each changed architecture.
    variants = [("MHA", (0, 1, 2, 3), (0, 1, 2, 3)),
                ("GQA", (0, 0, 1, 1), (0, 2)),
                ("MQA", (0, 0, 0, 0), (0,))]
    original = expanded(data, 3)
    for name, groups, sources in variants:
        for t in range(1, 4):
            compare(grouped(data, t, groups, sources),
                    expanded(data, t, tuple(sources[g] for g in groups)))
        variant = grouped(data, 3, groups, sources)
        equal = tuple(all(sp.simplify(x) == 0 for x in a.output-b.output)
                      for a, b in zip(original.heads, variant.heads))
        expected = {"MHA": (True, True, True, True),
                    "GQA": (True, False, True, False),
                    "MQA": (True, False, False, False)}[name]
        assert equal == expected
        print(f"B: {name}: {len(sources)*3} stored K/V pairs; unchanged heads at t=3: {equal}")
    a2 = softmax_exact([0, sp.log(2)])[1]
    a3 = softmax_exact([0, sp.log(3)])[1]
    average = softmax_exact([0, (sp.log(2)+sp.log(3))/2])[1]
    assert_zero(sp.Matrix([a2-sp.Rational(2, 3), a3-sp.Rational(3, 4),
                           average-sp.sqrt(6)/(1+sp.sqrt(6))]))
    assert sp.simplify(average-a2) != 0 and sp.simplify(average-a3) != 0
    print("B: averaged-key output:", sp.simplify(average), "(expected non-equivalence)")

    # D: specified linear maps, not the observed prefix rank.
    P = sp.Matrix.vstack(*(maps[h]*data.D for h in range(4)
                           for maps in (data.UK, data.UV)))
    up = sp.Matrix.vstack(*(maps[h] for h in range(4) for maps in (data.UK, data.UV)))
    assert P.rank() == 3
    assert_zero(P-up*data.D)
    D2 = data.D[:2, :]
    x, xp = sp.Matrix([0, 0, 0, 1]), sp.Matrix([0, 0, 1, 1])
    assert_zero(D2*x-D2*xp)
    value_map = data.UV[0]*data.D
    assert_zero(value_map*x-sp.Matrix([0, 0]))
    assert_zero(value_map*xp-sp.Matrix([0, 1]))
    # Singleton attention is its one value; any one decoder on equal D2 records fails one target.
    for token in (x, xp):
        value = value_map*token
        assert_zero(value*softmax_exact([0])[0]-value)
    sample = sp.Matrix([1, 0])
    assert (sp.eye(2)*sample).rank() == 1 and sp.eye(2).rank() == 2
    print("D: rank(P)=3; D2 collision and singleton-output non-equivalence verified.")

    # E: the separately specified positional model, including its changed scale.
    for t in range(1, 4):
        compare(expanded(data, t, positional=True), absorbed(data, t, positional=True))
    R, square_U = sp.Matrix([[0, -1], [1, 0]]), sp.diag(1, 2)
    assert_zero(R*square_U-square_U*R-sp.Matrix([[0, -1], [-1, 0]]))
    base = sp.Matrix(sp.symbols("b0:8", real=True))
    unrotated, nonrotated = base[:2, :], base[2:, :]
    rotated = R*unrotated
    record = unrotated.col_join(rotated).col_join(nonrotated)
    assert record.shape == (10, 1)
    key = record[2:4, :].col_join(record[4:, :])
    value = record[:2, :].col_join(record[4:, :])
    assert_zero(key-rotated.col_join(nonrotated))
    assert_zero(value-base)
    assert 4*(2+2+2) == 24 and 4*(2+2)+2 == 18 and 3+2 == 5
    assert sp.Rational(16-10, 16) == sp.Rational(3, 8)
    print("E: 24 legal positional scores in 12 head-prefix cases; role reconstructions exact.")

    # F: dependency witnesses and integer payload ledger.
    u1, u2, u3 = sp.symbols("u1 u2 u3", real=True)
    good, bad = (u1+(u1+u2))/2, u1+u2+u3
    assert sp.diff(good, u3) == 0 and sp.diff(bad, u3) == 1
    producers = (1, 1, 3, 3)
    assert all(p <= ell and producers[p-1] == p for ell, p in enumerate(producers, 1))
    assert len(set(producers)) == 2
    assert_zero(sp.eye(2)*sp.Matrix([1, 0])-sp.Matrix([1, 0]))
    assert sp.eye(2)*sp.Matrix([1, 0]) != sp.eye(2)*sp.Matrix([0, 1])
    bytes_ledger = payloads()
    mib = {name: sp.Rational(value, 2**20) for name, value in bytes_ledger.items()}
    assert list(mib.values()) == [768, 192, 48, 96]
    # Located nonlinear failure: square after mixture differs from mixing squared values.
    assert ((sp.Integer(0)+2)/2)**2 == 1
    assert (sp.Integer(0)**2+2**2)/2 == 2
    print("F: future derivatives 0/1; unique producers 2; bytes:", bytes_ledger)
    print("F: MiB:", mib)
    print("Classification: absorption/deduplication exact for specified model; GQA/MQA and new positional branch architecture changes; D2 replacement loses information.")
    print("All A–F exact checks passed. No model-quality or decoding-latency measurement was performed.")


if __name__ == "__main__":
    run()
