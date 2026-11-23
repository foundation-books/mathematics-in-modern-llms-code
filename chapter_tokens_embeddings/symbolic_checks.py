"""Executable instructor reference for Homework 4, Parts A-F.

Run from the repository root:
    python3 code/chapter_tokens_embeddings/symbolic_checks.py
Expected values are frozen separately, not computed by the rotary helper.
"""

from collections import Counter

import sympy as sp

from reference_inputs import (
    DECODED_TEXT, E, FULL_SCORE, GAMMA, GRADIENT_EXPECTED, GX,
    H_EXPECTED, IDS, OMEGA, PARTIAL_SCORE, POSITIONS,
    RESCALED_SCORE, S, VOCABULARY, W,
)


def rotation(phi):
    """Standard column rotation; the declared row action uses the transpose."""
    return sp.Matrix([[sp.cos(phi), -sp.sin(phi)],
                      [sp.sin(phi), sp.cos(phi)]])


def certify_zero_matrix(residual, label):
    reduced = residual.applyfunc(lambda entry: sp.simplify(sp.trigsimp(entry)))
    if reduced != sp.zeros(reduced.rows, reduced.cols):
        raise AssertionError(f"{label}: residual not certified zero:\n{reduced}")


def certify_equal(actual, expected, label):
    certify_zero_matrix(sp.Matrix([[actual - expected]]), label)


def rotary(position, partial=False):
    first = sp.eye(2) if partial else rotation(position * OMEGA[0])
    return sp.diag(first, rotation(position * OMEGA[1]))


def score(q, k, m, n, *, partial=False, scale=sp.Integer(1), opposite=False):
    if q.shape != (1, 4) or k.shape != (1, 4):
        raise ValueError("fixture rows must have shape 1 x 4")
    scale = sp.sympify(scale)
    if scale.is_positive is not True:
        raise ValueError("scale must be positive")
    qm, kn = rotary(sp.sympify(m) / scale, partial), rotary(sp.sympify(n) / scale, partial)
    qr, kr = (q * qm, k * kn) if opposite else (q * qm.T, k * kn.T)
    return sp.simplify(sp.trigsimp((qr * kr.T)[0]))


def pair_counts(corpus):
    return Counter(pair for sequence in corpus for pair in zip(sequence, sequence[1:]))


def merge(sequence, pair):
    result, index = [], 0
    while index < len(sequence):
        if tuple(sequence[index:index + 2]) == pair:
            result.append("".join(pair))
            index += 2
        else:
            result.append(sequence[index])
            index += 1
    return result


def masked_softmax(scores, mask):
    if scores.shape != mask.shape:
        raise ValueError("scores and mask must have equal shapes")
    result = sp.zeros(*scores.shape)
    for i in range(scores.rows):
        weights = [mask[i, j] * sp.exp(scores[i, j]) for j in range(scores.cols)]
        denominator = sum(weights)
        if denominator == 0:
            raise ValueError("each query needs an allowed key")
        for j, value in enumerate(weights):
            result[i, j] = value / denominator
    return result


def main():
    print("SymPy version:", sp.__version__)
    # A: shapes, row lookup, and exact logits; tokenizer/count checks also
    # independently cover Exercises 4.1-4.4 and the section 4.1 merge trace.
    assert VOCABULARY == ("A", "B", "C", "<eos>", "D", "E", "F", "G")
    assert "".join(DECODED_TEXT[i] for i in IDS) == "abac"
    assert "".join(DECODED_TEXT[i] for i in (4, 6)) == "abac"
    corpus = [list(text) for text in ("abac", "abac", "ab")]
    assert pair_counts(corpus) == {("a", "b"): 3, ("b", "a"): 2, ("a", "c"): 2}
    after_ab = [merge(seq, ("a", "b")) for seq in corpus]
    assert pair_counts(after_ab) == {("ab", "a"): 2, ("a", "c"): 2}
    assert merge(after_ab[0], ("a", "c")) == ["ab", "ac"]
    costs = [(v * 4, t * v * 4, t * t, t * (t + 1) // 2) for v, t in ((8, 4), (16, 3))]
    assert costs == [(32, 128, 16, 10), (64, 192, 9, 6)]
    X = S * E
    direct = sp.Matrix.vstack(*(E[i, :] for i in IDS))
    assert X == direct
    H, Z = X * W, X * W * E.T
    for matrix, shape in ((S, (4, 8)), (E, (8, 4)), (W, (4, 4)),
                          (GAMMA, (4, 8)), (X, (4, 4)), (H, (4, 4)), (Z, (4, 8))):
        assert matrix.shape == shape
    assert H == H_EXPECTED
    assert sp.Matrix([[1, 1, 0, 1]]) * E.T == sp.Matrix([[1, 2, 2, 1, 1, 1, 0, 1]])
    lookup_gradient = S.T * GX
    assert lookup_gradient == sp.Matrix([
        [4, 2, 0, 1], [0, 1, 1, 0], [0, 0, 2, 1],
        *([[0, 0, 0, 0]] * 5),
    ])
    print("A: X =", X)
    print("A: H =", H)
    print("A: Z =", Z)
    print("A: shapes and lookup certified; cost and merge fixtures certified")

    # B: directly differentiate 32 independent real variables before any
    # numerical substitution. This oracle does not call the path formula.
    variables = sp.symbols("e0:32", real=True)
    symbolic_E = sp.Matrix(8, 4, variables)
    symbolic_Z = S * symbolic_E * W * symbolic_E.T
    loss = sum(GAMMA[i, j] * symbolic_Z[i, j] for i in range(4) for j in range(8))
    direct_gradient = sp.Matrix(8, 4, [sp.diff(loss, variable) for variable in variables])
    input_path = S.T * GAMMA * symbolic_E * W.T
    output_path = GAMMA.T * S * symbolic_E * W
    certify_zero_matrix(input_path + output_path - direct_gradient, "B tied gradient")
    substitutions = dict(zip(variables, list(E)))
    total = direct_gradient.subs(substitutions)
    assert total == GRADIENT_EXPECTED
    assert loss.subs(substitutions) == 6
    assert input_path.subs(substitutions)[4, :] == sp.zeros(1, 4)
    assert output_path.subs(substitutions)[4, :] == sp.Matrix([[1, 2, 0, 0]])
    print("B: scalar probe = 6; symbolic residual is zero; gradient =", total)

    # C: arbitrary real angles/coordinates, arbitrary content, positive
    # frequencies. Blockwise matrix residuals certify the score for all rows.
    alpha, beta = sp.symbols("alpha beta", real=True)
    m, n, shift = sp.symbols("m n shift", real=True)
    omega0, omega1 = sp.symbols("omega0 omega1", positive=True)
    certify_zero_matrix(rotation(alpha).T * rotation(alpha) - sp.eye(2), "C orthogonality")
    certify_zero_matrix(rotation(alpha).T - rotation(-alpha), "C transpose")
    certify_zero_matrix(rotation(alpha) * rotation(beta) - rotation(alpha + beta), "C composition")
    block = lambda position: sp.diag(rotation(position * omega0), rotation(position * omega1))
    certify_zero_matrix(block(m).T * block(n) - block(n - m), "C relative block")
    certify_zero_matrix(block(m + shift).T * block(n + shift) - block(n - m), "C common shift")
    certify_zero_matrix(block(m).T * block(m) - sp.eye(4), "C block norm")
    a, b, c, d = sp.symbols("a b c d", real=True)
    delta, omega = sp.symbols("delta omega", real=True)
    expanded = (sp.Matrix([[a, b]]) * rotation(delta * omega) * sp.Matrix([c, d]))[0]
    expected = (a * c + b * d) * sp.cos(delta * omega) + (b * c - a * d) * sp.sin(delta * omega)
    certify_equal(expanded, expected, "C sine coefficient")
    print("C: arbitrary-angle identities and two-pair relative/common-shift identities certified")

    # Exercise 4.5: the scalar causal example and the covariant mask.
    swap = sp.Matrix([[0, 1], [1, 0]])
    causal = sp.Matrix([[1, 0], [1, 1]])
    values = sp.Matrix([0, 1])
    zero_scores = sp.zeros(2, 2)
    causal_output = masked_softmax(zero_scores, causal) * values
    swapped_input_output = masked_softmax(zero_scores, causal) * swap * values
    assert causal_output == sp.Matrix([0, sp.Rational(1, 2)])
    assert swapped_input_output == sp.Matrix([1, sp.Rational(1, 2)])
    assert swap * causal_output == sp.Matrix([sp.Rational(1, 2), 0])
    assert swapped_input_output != swap * causal_output
    assert masked_softmax(zero_scores, swap * causal * swap.T) * swap * values == swap * causal_output

    # D: independent exact expectations; all transformed rows keep width 4.
    q, k = X[2, :], X[1, :]
    assert q.shape == k.shape == (1, 4)
    assert 2 + 2 == q.cols
    certify_equal((q * k.T)[0], 0, "D unrotated")
    certify_equal(score(q, k, 2, 1), FULL_SCORE, "D full")
    certify_equal(score(q, k, 2, 1, partial=True), PARTIAL_SCORE, "D partial")
    certify_equal(score(q, k, 7, 6), FULL_SCORE, "D common shift")
    certify_equal(score(q, k, 7, 6, partial=True), PARTIAL_SCORE, "D partial common shift")
    for position, row in ((2, q), (1, k)):
        for partial in (False, True):
            transformed = row * rotary(position, partial).T
            assert transformed.shape == (1, 4)
            certify_equal((transformed * transformed.T)[0], (row * row.T)[0], "D norm")
    unrotated_part = (q[:, :2] * k[:, :2].T)[0]
    rotated_part = (q[:, 2:] * rotation((1 - 2) * OMEGA[1]) * k[:, 2:].T)[0]
    certify_equal(score(q, k, 2, 1, partial=True), unrotated_part + rotated_part, "D decomposition")
    print("D: full =", FULL_SCORE, "; partial =", PARTIAL_SCORE, "; shift unchanged")

    # E: symbolic phase proof and exact endpoint/periodic witnesses.
    scale = sp.symbols("scale", positive=True)
    certify_equal((n / scale - m / scale) * omega, (n - m) * omega / scale, "E phase")
    ids_before = tuple(IDS)
    certify_equal(score(q, k, 2, 1, scale=sp.Integer(2)), RESCALED_SCORE, "E rescaled")
    assert ids_before == IDS and len(IDS) == 4 and POSITIONS == (0, 1, 2, 3)
    endpoint_scale = sp.Rational(9 - 1, 5 - 1)
    assert endpoint_scale == 2 and 0 / endpoint_scale == 0 and 8 / endpoint_scale == 4
    periodic = [sp.cos(index * sp.pi / 2) for index in range(5)]
    assert periodic == [1, 0, -1, 0, 1]
    print("E: rescaled =", RESCALED_SCORE, "; endpoints 0->0, 8->4; periodic =", periodic)


    # Part E extension: independent learned frequencies, with the printed row action.
    a, b, c, d, delta, alpha = sp.symbols("a b c d delta alpha", real=True)
    reference = sp.symbols("reference_rate", positive=True)
    learned = reference * sp.exp(alpha)
    angle = delta * learned
    row_score = (sp.Matrix([[a, b]]) * rotation(angle) * sp.Matrix([c, d]))[0]
    expansion = (a*c + b*d)*sp.cos(angle) + (b*c - a*d)*sp.sin(angle)
    certify_equal(row_score, expansion, "E learned row convention")
    derivative = delta*learned*(-(a*c + b*d)*sp.sin(angle)
                               + (b*c - a*d)*sp.cos(angle))
    certify_equal(sp.diff(row_score, alpha), derivative, "E local log-frequency derivative")
    certify_equal(learned.subs(alpha, 0), reference, "E learned initialization")
    # Pure matrix rotations at the three independently specified spectra.
    unit = sp.Matrix([[1, 0, 0, 0]])
    spectra = ((sp.pi/4, sp.pi/8), (sp.pi/8, sp.pi/16), (sp.pi/2, sp.pi/8))
    targets = (sp.sqrt(2)/2, sp.cos(sp.pi/8), 0)
    for rates, target in zip(spectra, targets):
        def transformed(position):
            return unit * sp.diag(*(rotation(position*rate) for rate in rates)).T
        for query_pos, key_pos in ((2, 1), (7, 6)):
            query, key = transformed(query_pos), transformed(key_pos)
            certify_equal((query*key.T)[0], target, "E learned score and common shift")
            certify_equal(query.dot(query), 1, "E learned query norm")
            certify_equal(key.dot(key), 1, "E learned key norm")
    at_learned = {a: 1, b: 0, c: 1, d: 0, delta: -1,
                  reference: sp.pi/4, alpha: sp.log(2)}
    certify_equal(sp.diff(row_score, alpha).subs(at_learned), -sp.pi/2,
                  "E learned-point derivative")
    certify_equal(derivative.subs(at_learned | {delta: -2}), 0,
                  "E distance does not force nonzero sensitivity")
    assert spectra[2][0]/spectra[0][0] == 2 and spectra[2][1]/spectra[0][1] == 1
    print("E learned: scores sqrt(2)/2, cos(pi/8), 0; derivative -pi/2; invariants passed")

    # Exercise 3.8: a base change leaves the zeroth frequency unchanged.
    base, changed_base = sp.symbols("base changed_base", positive=True)
    ratios = [(changed_base / base)**(-sp.Rational(2*j, 4))
              for j in (0, 1)]
    assert ratios[0] == 1 and ratios[0] != sp.Rational(1, 2)
    certify_equal(ratios[1].subs(changed_base, 4 * base),
                  sp.Rational(1, 2), "4.10 second-pair base ratio")
    print("4.10: uniform scaling gives (1/2,1/2); base change gives", ratios)

    # F: two separate faults. Norms pass for both. An opposite but consistent
    # convention also passes common-shift tests; the independent score fails.
    opposite = score(q, k, 2, 1, opposite=True)
    reset = score(q, k, 0, 1)
    certify_equal(opposite, -FULL_SCORE, "F opposite expected discrepancy")
    certify_equal(reset, -FULL_SCORE, "F reset expected discrepancy")
    certify_equal(score(q, k, 7, 6, opposite=True), opposite, "F opposite shift still passes")
    for position, row in ((2, q), (1, k)):
        transformed = row * rotary(position)
        certify_equal((transformed * transformed.T)[0], (row * row.T)[0], "F opposite norm")
    reset_query = q * rotary(0).T
    certify_equal((reset_query * reset_query.T)[0], (q * q.T)[0], "F reset norm")
    reset_shifted = score(q, k, 0, 6)  # Fault continues to reset A's coordinate.
    assert sp.simplify(reset_shifted - reset) != 0
    assert sp.simplify(opposite - FULL_SCORE) != 0
    assert sp.simplify(reset - FULL_SCORE) != 0
    print("F: variant | exact score | norm | common shift | reference sign/index")
    print("F: intended |", FULL_SCORE, "| pass | pass | pass")
    print("F: opposite |", opposite, "| pass | pass | fail")
    print("F: reset query |", reset, "| pass | fail | fail")
    print("All Parts A-F and the exact exercise fixtures passed.")


if __name__ == "__main__":
    main()
