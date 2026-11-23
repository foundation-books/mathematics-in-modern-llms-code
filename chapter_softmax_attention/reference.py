"""Instructor reference: exact identities and numerical checks, not benchmarks.

Run from the repository root with Python 3.12.3 and SymPy 1.13.3.
The expected constants are assertion targets; all outputs are computed.
"""

import math
import sys

import sympy as sp

from student_setup import K, V, half, q, s, target, tau


def equal(actual, expected):
    """Require an exact zero residual, entry by entry for matrices."""
    if isinstance(actual, sp.MatrixBase):
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        residual = (actual - expected).applyfunc(sp.simplify)
        assert residual == sp.zeros(*actual.shape), residual
    else:
        residual = sp.simplify(actual - expected)
        assert residual == 0, residual


def expected_failure(actual, expected, name):
    """Run the usual equality check on a deliberately wrong implementation."""
    try:
        equal(actual, expected)
    except AssertionError:
        print("Expected failing check:", name)
    else:
        raise AssertionError("negative test unexpectedly passed: " + name)


def masked_softmax(scores, temperature=sp.Integer(1), allowed=None):
    """Column scores; allowed is a nonempty set of zero-based integer indices."""
    scores = sp.Matrix(scores)
    if scores.cols != 1 or scores.rows < 1:
        raise ValueError("scores must be a nonempty column")
    temperature = sp.sympify(temperature)
    if temperature.is_positive is not True:
        raise ValueError("temperature must be known positive")
    if any(x.is_real is not True or x.is_finite is not True for x in scores):
        raise ValueError("scores must be known finite real numbers")
    if allowed is None:
        allowed = tuple(range(scores.rows))
    else:
        allowed = tuple(allowed)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("support must be nonempty with unique indices")
    if any(type(j) is not int or not 0 <= j < scores.rows for j in allowed):
        raise ValueError("support index out of range")
    denominator = sum(sp.exp(scores[j] / temperature) for j in allowed)
    return sp.Matrix([
        sp.exp(x / temperature) / denominator if j in allowed else 0
        for j, x in enumerate(scores)
    ])


def stable_float_softmax(logits):
    """Finite Python float logits; max shifting in the logit domain."""
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("a nonempty finite logit vector is required")
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    denominator = math.fsum(weights)
    return [x / denominator for x in weights]


def matrix_directional_check():
    """Check every shared-X branch against a direct symbolic directional derivative."""
    Q = sp.Matrix.vstack(q.T, q.T, q.T)
    A = sp.Matrix.vstack(*[
        masked_softmax(K * q / sp.sqrt(2), allowed=tuple(range(i))).T
        for i in (1, 2, 3)
    ])
    GO = sp.Matrix([[1, 2], [-1, 1], [2, -1]])
    GV = A.T * GO
    GA = GO * V.T
    c = GA.multiply_elementwise(A) * sp.ones(3, 1)
    GS = A.multiply_elementwise(GA - c * sp.ones(1, 3))
    GQ = GS * K / sp.sqrt(2)
    GK = GS.T * Q / sp.sqrt(2)
    GX = GQ * Q.T + GK * K.T + GV * V.T  # X=I; W_Q=Q, W_K=K, W_V=V.
    equal(GS * sp.ones(3, 1), sp.zeros(3, 1))
    assert GS[0, 1] == 0 and GS[0, 2] == 0 and GS[1, 2] == 0
    D = sp.Matrix([[1, -1, 2], [2, 1, -1], [-1, 3, 1]]) / 7
    t = sp.symbols("direction_step", real=True)
    X = sp.eye(3) + t * D
    scores = X * Q * (X * K).T / sp.sqrt(2)
    At = sp.Matrix.vstack(*[
        masked_softmax(scores[i, :].T, allowed=tuple(range(i + 1))).T
        for i in range(3)
    ])
    Ot = At * X * V
    linear_loss = sum(GO.multiply_elementwise(Ot))
    direct = sp.diff(linear_loss, t).subs(t, 0)
    predicted = sum(GX.multiply_elementwise(D))
    equal(direct, predicted)
    # This direction gives nonzero matching and value contributions.
    assert sp.simplify(sum((GQ * Q.T).multiply_elementwise(D))) != 0
    assert sp.simplify(sum((GK * K.T).multiply_elementwise(D))) != 0
    assert sp.simplify(sum((GV * V.T).multiply_elementwise(D))) != 0
    return direct


def learned_scale_checks():
    """Part B extension: finite scores are scaled before support restriction."""
    scores = sp.Matrix(sp.symbols("scale_score0:3", real=True))
    multiplier = sp.symbols("score_multiplier", real=True)
    weights = masked_softmax(multiplier * scores)
    partition = sum(sp.exp(multiplier*x) for x in scores)
    log_weights = [multiplier*x-sp.log(partition) for x in scores]
    for weight, log_weight in zip(weights, log_weights):
        equal(sp.expand_log(sp.log(weight)), log_weight)
    entropy = -sum(w*lw for w, lw in zip(weights, log_weights))
    mean = weights.dot(scores)
    variance = sum(w*(x-mean)**2 for w, x in zip(weights, scores))
    equal(sp.diff(entropy, multiplier), -multiplier*variance)
    equal(weights.subs(multiplier, 0), sp.ones(3, 1)/3)
    equal(weights.subs(multiplier, 1), masked_softmax(scores))
    equal(sp.diff(entropy, multiplier).subs(dict.fromkeys(scores, 2)), 0)
    fixed = dict(zip(scores, (0, 1, 2)))
    for value, sign in ((1, -1), (0, 0), (-1, 1)):
        slope = sp.simplify((-multiplier*variance).subs(fixed).subs(multiplier, value))
        assert sp.sign(slope) == sign
    u, v = sp.symbols("u v", real=True)
    position = sp.symbols("position", nonnegative=True)
    scale = u*sp.log(position+1)+v
    equal(scale.subs(position, 0), v)
    equal(scale.subs({u: 0, v: 1}), 1)
    # Sparse support has two keys even when position+1 is sixteen.
    sparse = masked_softmax(scale.subs({position: 15, u: 1, v: 0})*sp.Matrix([1, 0, 9]),
                            allowed=(0, 1))
    equal(sparse, sp.Matrix([sp.Rational(16, 17), sp.Rational(1, 17), 0]))
    n = sp.symbols("source_count", integer=True, positive=True)
    gap = sp.symbols("target_gap", positive=True)
    target = 1/(1+(n-1)*sp.exp(-(u*sp.log(n)+v)*gap))
    exponential = sp.expand_power_exp(sp.exp(sp.expand(-(u*sp.log(n)+v)*gap)))
    equal(exponential, sp.exp(-v*gap)*n**(-u*gap))
    equal(target, 1/(1+(n-1)*exponential))
    compensated = sp.simplify(target.subs({gap: 1, u: 1, v: 0}))
    equal(compensated.subs(n, 4), sp.Rational(4, 7))
    equal(compensated.subs(n, 16), sp.Rational(16, 31))
    for exponent, expected in ((sp.Rational(1, 2), 0), (1, 1/(1+sp.exp(-v))), (2, 1)):
        equal(sp.limit(target.subs({gap: 1, u: exponent}), n, sp.oo), expected)
    # General three-regime proof: the denominator's nonconstant term is
    # exp(-vD)*(1-1/n)*n**(1-uD); the final power has sign-determined limit.
    positive_power = sp.symbols("positive_power", positive=True)
    equal(sp.limit(n**(-positive_power), n, sp.oo), 0)
    assert sp.limit(n**positive_power, n, sp.oo) == sp.oo
    for count in (4, 16):
        fixed_scale = 1/(1+(count-1)*sp.exp(-1))
        equal(fixed_scale, sp.E/(sp.E+count-1))
    print("B learned scale: symbolic entropy, sign/equal-score cases, support and three limits passed")


def main():
    print("Python:", sys.version.split()[0])
    print("SymPy:", sp.__version__)
    print("Python float significand:", sys.float_info.mant_dig, "bits")
    scores0 = K * q / sp.sqrt(2)
    a_star = masked_softmax(scores0)
    o0 = V.T * a_star
    fixture = dict(zip(s, scores0)) | {tau: 1}
    equal(scores0, sp.Matrix([0, sp.log(2), sp.log(3)]))
    equal(a_star, sp.Matrix([sp.Rational(1, 6), sp.Rational(1, 3), half]))
    equal(o0, sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 6)]))
    restricted = masked_softmax(scores0, allowed=(0, 1))
    equal(restricted, sp.Matrix([sp.Rational(1, 3), sp.Rational(2, 3), 0]))
    equal(V.T * restricted, sp.Matrix([sp.Rational(1, 3), sp.Rational(2, 3)]))
    wrong_mask = sp.Matrix([a_star[0], a_star[1], 0])
    equal(sum(wrong_mask), half)
    assert wrong_mask != restricted
    equal(V.T * wrong_mask, sp.Matrix([sp.Rational(1, 6), sp.Rational(1, 3)]))
    for bad_support in ((), (0, 0), (-1,), (3,)):
        try:
            masked_softmax(scores0, allowed=bad_support)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid support accepted")
    for bad_tau in (0, -1):
        try:
            masked_softmax(scores0, temperature=bad_tau)
        except ValueError:
            pass
        else:
            raise AssertionError("nonpositive temperature accepted")
    causal_A = sp.Matrix.vstack(*[
        masked_softmax(scores0, allowed=tuple(range(i))).T
        for i in (1, 2, 3)
    ])
    equal(causal_A * V, sp.Matrix([
        [1, 0], [sp.Rational(1, 3), sp.Rational(2, 3)],
        [sp.Rational(2, 3), sp.Rational(5, 6)],
    ]))
    equal(masked_softmax(sp.zeros(3, 1)), sp.ones(3, 1) / 3)
    equal(masked_softmax(sp.Matrix([7])), sp.Matrix([1]))
    print("A: exact forward, proper mask, causal matrix, and domain checks passed")

    a = masked_softmax(s, tau)
    Z = sum(sp.exp(x / tau) for x in s)
    F = tau * sp.log(Z)
    J = a.jacobian(s)
    J_formula = (sp.diag(*a) - a * a.T) / tau
    equal(J, J_formula)
    equal(sp.Matrix([sp.diff(F, x) for x in s]), a)
    equal(sp.hessian(F, tuple(s)), J)
    equal(J * sp.ones(3, 1), sp.zeros(3, 1))
    J0 = J.subs(fixture).applyfunc(sp.simplify)
    equal(J0, sp.Matrix([
        [sp.Rational(5, 36), -sp.Rational(1, 18), -sp.Rational(1, 12)],
        [-sp.Rational(1, 18), sp.Rational(2, 9), -sp.Rational(1, 6)],
        [-sp.Rational(1, 12), -sp.Rational(1, 6), sp.Rational(1, 4)],
    ]))
    equal(sp.diff(masked_softmax(sp.Matrix([s[0]]), tau)[0], s[0]), 0)
    restricted_symbolic = masked_softmax(s, tau, allowed=(0, 1))
    equal(restricted_symbolic.jacobian(s)[:, 2], sp.zeros(3, 1))
    print("B: three-source Jacobian/Hessian identities and masked derivative passed")

    u = sp.Matrix(sp.symbols("u1 u2 u3", positive=True))
    w = u / sum(u)
    H = -sum(x * sp.log(x) for x in w)
    # Positive real p permits this log rewrite without complex branch changes.
    log_a = [s[j] / tau - sp.log(Z) for j in range(3)]
    for j in range(3):
        equal(sp.expand_log(sp.log(a[j])), log_a[j])
    KL = sum(w[j] * (sp.log(w[j]) - log_a[j]) for j in range(3))
    equal(F - ((w.T * s)[0] + tau * H), tau * KL)
    uniform = sp.ones(3, 1) / 3
    gap = F.subs(fixture) - (uniform.dot(scores0) + sp.log(3))
    equal(gap, sp.log(2) - sp.log(6) / 3)
    assert gap.evalf(30) > 0
    equal(sp.limit(-u[0] * sp.log(u[0]), u[0], 0, dir="+"), 0)
    print("C: constrained interior KL gap, uniform gap, and boundary limit passed")

    output = V.T * a
    g = output - target
    loss = half * g.dot(g)
    grad_s = sp.Matrix([sp.diff(loss, x) for x in s])
    grad_formula = sp.Matrix([
        a[j] / tau * g.dot(V[j, :].T - output) for j in range(3)
    ])
    equal(grad_s, grad_formula)
    grad0 = grad_s.subs(fixture).applyfunc(sp.simplify)
    g0 = o0 - target
    equal(loss.subs(fixture), sp.Rational(29, 72))
    equal(grad0, sp.Matrix([-sp.Rational(29, 216), sp.Rational(13, 108),
                            sp.Rational(1, 72)]))
    equal(sum(grad0), 0)
    Gq = K.T * grad0 / sp.sqrt(2)
    GK = grad0 * q.T / sp.sqrt(2)
    GV = a_star * g0.T
    equal(Gq, sp.Matrix([(sp.Rational(13, 108) * sp.log(2)
                         + sp.log(3) / 72) / sp.sqrt(2), 0]))
    equal(GK, sp.Matrix([[-sp.Rational(29, 216), 0],
                         [sp.Rational(13, 108), 0], [sp.Rational(1, 72), 0]]))
    equal(GV, sp.Matrix([[-sp.Rational(1, 18), sp.Rational(5, 36)],
                         [-sp.Rational(1, 9), sp.Rational(5, 18)],
                         [-sp.Rational(1, 6), sp.Rational(5, 12)]]))
    qx = sp.Matrix(sp.symbols("q1 q2", real=True))
    Kx = sp.Matrix(3, 2, sp.symbols("k0:6", real=True))
    Vx = sp.Matrix(3, 2, sp.symbols("v0:6", real=True))
    ax = masked_softmax(Kx * qx / sp.sqrt(2))
    ox = Vx.T * ax
    Lx = half * (ox - target).dot(ox - target)
    local_subs = dict(zip(qx, q)) | dict(zip(Kx, K)) | dict(zip(Vx, V))
    equal(sp.Matrix([sp.diff(Lx, x).subs(local_subs) for x in qx]), Gq)
    equal(sp.Matrix(3, 2, [sp.diff(Lx, x).subs(local_subs) for x in Kx]), GK)
    equal(sp.Matrix(3, 2, [sp.diff(Lx, x).subs(local_subs) for x in Vx]), GV)
    print("D: score/value/query/key derivatives computed independently and passed")
    print("Central differences: 50 decimal digits; columns are absolute errors")
    for step in (sp.Rational(1, 100), sp.Rational(1, 10000), sp.Rational(1, 1000000)):
        errors = []
        for j in range(3):
            plus = fixture | {s[j]: scores0[j] + step}
            minus = fixture | {s[j]: scores0[j] - step}
            difference = (loss.subs(plus).evalf(50) - loss.subs(minus).evalf(50))
            approximation = difference / (2 * step)
            errors.append(abs(approximation - grad0[j].evalf(50)))
        print(str(step), *(str(x.evalf(8)) for x in errors))

    shift = sp.symbols("b", real=True)
    equal(masked_softmax(s + shift * sp.ones(3, 1), tau), a)
    equal(masked_softmax(scores0 + 1000 * sp.ones(3, 1)), a_star)
    equal(F.subs(dict(zip(s, s + shift * sp.ones(3, 1))), simultaneous=True), F + shift)
    correct_tau2 = J.subs(fixture | {tau: 2})
    missing_tau = (sp.diag(*a) - a * a.T).subs(fixture | {tau: 2})
    assert sp.simplify((missing_tau - correct_tau2)[0, 0]) != 0
    expected_failure(missing_tau, correct_tau2, "missing 1/tau at tau=2")
    expected_failure(wrong_mask, restricted, "post-softmax zeroing without renormalization")
    equal(missing_tau / 2, correct_tau2)
    equal(wrong_mask / sum(wrong_mask), restricted)
    print("E: both deliberate bugs detected; factor and renormalization repairs passed")
    try:
        math.exp(1000)
    except OverflowError:
        print("Python math.exp(1000): OverflowError on this bounded-float backend")
    else:
        raise AssertionError("expected bounded-float overflow was not observed")
    for logits in ([0.0, 1.0, 2.0], [1000.0, 1001.0, 1002.0]):
        stable = stable_float_softmax(logits)
        exact = masked_softmax(sp.Matrix([0, 1, 2]))
        assert all(abs(x - float(y.evalf(50))) < 1e-14 for x, y in zip(stable, exact))
        print("stable Python floats:", stable)
    # This fixture shift was applied to scores; at tau=1 they are also logits.
    stable0 = stable_float_softmax([float(x.evalf()) + 1000 for x in scores0])
    assert all(abs(x - float(y)) < 1e-12 for x, y in zip(stable0, a_star))
    print("high-precision SymPy exp(1000) is finite:", sp.exp(1000).evalf(8))

    Vdup = sp.Matrix([[1, 0], [1, 0], [0, 1]])
    a_first = sp.Matrix([half, sp.Rational(1, 4), sp.Rational(1, 4)])
    a_second = sp.Matrix([sp.Rational(1, 4), half, sp.Rational(1, 4)])
    assert a_first != a_second
    equal(Vdup.T * a_first, sp.Matrix([sp.Rational(3, 4), sp.Rational(1, 4)]))
    equal(Vdup.T * a_first, Vdup.T * a_second)
    equal(masked_softmax(a_first.applyfunc(sp.log)), a_first)
    equal(masked_softmax(a_second.applyfunc(sp.log)), a_second)
    print("F: duplicate-value counterexample and finite realizing logits passed")

    head2 = V.T * masked_softmax(sp.Matrix([sp.log(3), sp.log(2), 0]))
    equal(head2, sp.Matrix([sp.Rational(2, 3), half]))
    WO = sp.Matrix.vstack(sp.eye(2), sp.diag(1, -1))
    mixed_row = sp.Matrix.hstack(o0.T, head2.T) * WO
    mixed_col = o0 + sp.diag(1, -1).T * head2
    equal(mixed_row.T, mixed_col)
    equal(mixed_col, sp.Matrix([sp.Rational(4, 3), sp.Rational(1, 3)]))
    for temperature in (half, sp.Integer(1), sp.Integer(2)):
        weights = masked_softmax(scores0, temperature).applyfunc(sp.simplify)
        print("temperature", temperature, "weights:", weights.T,
              "output:", (V.T * weights).T)
    tie = masked_softmax(sp.Matrix([1, 1, 0]), tau)
    equal(tie.applyfunc(lambda x: sp.limit(x, tau, 0, dir="+")),
          sp.Matrix([half, half, 0]))
    equal(a.applyfunc(lambda x: sp.limit(x, tau, sp.oo)), sp.ones(3, 1) / 3)
    print("Additional checkpoints: heads, temperature limits, and tie passed")
    learned_scale_checks()
    matrix_directional_check()
    print("Matrix/shared-X check: direct symbolic directional derivative passed")
    print("All exact residual assertions and numerical consistency checks passed.")
    print("Fixed-size checks supplement the general proofs in the chapter.")


if __name__ == "__main__":
    main()
