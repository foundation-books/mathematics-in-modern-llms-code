"""Executed instructor reference for the six Chapter 10 homework tasks.

This checks identities in dimension four and constructed diagnostic inputs.
General-width proofs and interpretations are in the manuscript/answer key.
Run from the repository root; no automatic differentiation framework is used.
"""

import argparse
import json
from pathlib import Path
import platform
import runpy

import sympy as sp

from chapter10_fixture import (
    P_lift, Q, Wd, Wg, Wu, core_norm, eps0, gated, c_lift, o_star, sigmoid, silu,
    v_dir, x0, zero_matrix_test,
)


def verify():
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def matrix_check(name, actual, expected):
        check(name, zero_matrix_test(actual - expected))

    def equal(name, actual, expected):
        check(name, sp.simplify(actual - expected) == 0)

    def rejects(name, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    # A: independent exact arithmetic, supplied data and lawful domains.
    matrix_check("A-lift", x0, c_lift + P_lift * o_star)
    check("A-shapes", x0.shape == (4, 1) and Wg.shape == Wu.shape == (4, 2)
          and Wd.shape == (2, 4))
    mu = sum(x0) / 4
    c0 = x0 - sp.ones(4, 1) * mu
    q0, variance0 = x0.dot(x0) / 4, c0.dot(c0) / 4
    equal("A-mean", mu, Q(3, 8))
    equal("A-mean-square", q0, Q(113, 144))
    equal("A-variance", variance0, Q(371, 576))
    equal("A-second-moment-decomposition", q0, variance0 + mu**2)
    for tag, epsilon, rad_r, rad_l in [("zero", 0, 113, 371),
                                      ("positive", eps0, 114, 375)]:
        r_value = core_norm(x0, epsilon)
        l_value = core_norm(x0, epsilon, True)
        matrix_check(f"A-R-{tag}", r_value,
                     sp.Matrix([8, 10, 12, -12]) / sp.sqrt(rad_r))
        matrix_check(f"A-L-{tag}", l_value,
                     sp.Matrix([7, 11, 15, -33]) / sp.sqrt(rad_l))
        equal(f"A-L-mean-{tag}", sum(l_value), 0)
        equal(f"A-R-moment-{tag}", r_value.dot(r_value) / 4,
              q0 / (q0 + epsilon))
        equal(f"A-L-moment-{tag}", l_value.dot(l_value) / 4,
              variance0 / (variance0 + epsilon))
    rejects("A-reject-zero-R", core_norm, sp.zeros(4, 1))
    rejects("A-reject-constant-L", core_norm, 7 * sp.ones(4, 1), center=True)
    rejects("A-reject-row", core_norm, x0.T)
    rejects("A-reject-empty", core_norm, sp.zeros(0, 1))
    rejects("A-reject-negative-eps", core_norm, x0, -1)
    rejects("A-reject-gate-shape", gated, sp.ones(3, 1))

    # B: differentiate independently before substituting the fixture.
    xs = sp.Matrix(sp.symbols("x1:5", real=True))
    eps = sp.symbols("epsilon", positive=True)
    ones, identity = sp.ones(4, 1), sp.eye(4)
    pc = identity - ones * ones.T / 4
    centered = pc * xs
    r = sp.sqrt(xs.dot(xs) / 4 + eps)
    s = sp.sqrt(centered.dot(centered) / 4 + eps)
    jr = identity / r - xs * xs.T / (4 * r**3)
    jl = pc / s - centered * centered.T / (4 * s**3)
    matrix_check("B-projector", pc * pc, pc)
    matrix_check("B-projector-symmetric", pc.T, pc)
    matrix_check("B-JR-independent", core_norm(xs, eps).jacobian(xs), jr)
    matrix_check("B-JL-independent", core_norm(xs, eps, True).jacobian(xs), jl)
    matrix_check("B-R-radial", jr * xs, eps * xs / r**3)
    matrix_check("B-L-shift", jl * ones, sp.zeros(4, 1))
    matrix_check("B-L-radial", jl * centered, eps * centered / s**3)
    fixture_sub = dict(zip(xs, x0))
    jr0 = jr.subs(fixture_sub).subs(eps, eps0)
    jl0 = jl.subs(fixture_sub).subs(eps, eps0)
    rank_table = {
        "R_zero": jr.subs(fixture_sub).subs(eps, 0).rank(),
        "R_positive": jr0.rank(),
        "L_zero": jl.subs(fixture_sub).subs(eps, 0).rank(),
        "L_positive": jl0.rank(),
    }
    check("B-four-ranks", list(rank_table.values()) == [3, 4, 2, 3])
    matrix_check("B-R-at-zero", jr.subs(dict(zip(xs, [0] * 4))),
                 identity / sp.sqrt(eps))
    matrix_check("B-L-at-constant", jl.subs(dict(zip(xs, [7] * 4))),
                 pc / sp.sqrt(eps))
    wr, wl = sp.Matrix([5, -4, 0, 0]), sp.Matrix([1, -2, 1, 0])
    matrix_check("B-R-transverse", jr0 * wr, 12 * wr / sp.sqrt(114))
    matrix_check("B-L-transverse", jl0 * wl, 24 * wl / sp.sqrt(375))
    check("B-gain-does-not-preserve-eigendirection",
          sp.diag(1, 2, 1, 1) * wr == sp.Matrix([5, -8, 0, 0]))

    # C: exact laws and exact nonzero witnesses, not unresolved truth values.
    matrix_check("C-L-shift", core_norm(x0 + 7 * ones, eps0, True),
                 core_norm(x0, eps0, True))
    for scale in [3, -3]:
        sign = sp.sign(scale)
        matrix_check(f"C-R-scale-{scale}", core_norm(scale * x0, eps0),
                     sign * core_norm(x0, eps0 / scale**2))
        matrix_check(f"C-L-scale-shift-{scale}",
                     core_norm(scale * x0 + 7 * ones, eps0, True),
                     sign * core_norm(x0, eps0 / scale**2, True))
    shifted = core_norm(x0 + 7 * ones, eps0)
    check("C-R-shift-fails", sp.simplify(shifted[0]**2
          - core_norm(x0, eps0)[0]**2) != 0)
    check("C-fixed-eps-scaling-fails", sp.simplify(core_norm(3*x0, eps0)[0]**2
          - core_norm(x0, eps0)[0]**2) != 0)
    matrix_check("C-constant-regularized-L", core_norm(7 * ones, eps0, True),
                 sp.zeros(4, 1))

    # D: independent z Jacobian, product-rule form and a one-parameter path.
    h_norm_symbols = sp.Matrix(sp.symbols("h_norm1:5", real=True))
    u_g, u_v = Wg.T * h_norm_symbols, Wu.T * h_norm_symbols
    phi = u_g.applyfunc(silu)
    phi_prime = u_g.applyfunc(lambda t: sigmoid(t) + t*sigmoid(t)*(1-sigmoid(t)))
    jg = gated(h_norm_symbols).jacobian(h_norm_symbols)
    gate_path = sp.diag(*u_v.multiply_elementwise(phi_prime)) * Wg.T
    value_path = sp.diag(*phi) * Wu.T
    hand = Wd.T * (gate_path + value_path)
    matrix_check("D-two-path-JG", jg, hand)
    gate_witness = dict(zip(h_norm_symbols, [0, 0, 1, 1]))
    check("D-held-fixed-gate-is-different",
          (jg - Wd.T * value_path).subs(gate_witness)[0, 0] == Q(1, 2))
    h_norm0 = core_norm(x0, eps0)
    u_g0, u_v0 = Wg.T * h_norm0, Wu.T * h_norm0
    p0 = u_g0.applyfunc(silu).multiply_elementwise(u_v0)
    b0, y0 = gated(h_norm0), x0 + gated(h_norm0)
    expected_p = sp.Matrix([Q(16, 19)*sigmoid(8/sp.sqrt(114)),
                           -Q(20, 19)*sigmoid(10/sp.sqrt(114))])
    matrix_check("D-exact-hidden-products", p0, expected_p)
    matrix_check("D-output-copies", b0, sp.Matrix([p0[0], p0[1], p0[0], p0[1]]))
    expected_b = ["0.571805743691", "-0.756220517536"] * 2
    check("D-rounded-forward", all(abs(b0[i].evalf(30)-sp.Float(v, 30)) < Q(1, 10**11)
          for i, v in enumerate(expected_b)))
    jg0 = jg.subs(dict(zip(h_norm_symbols, h_norm0)))
    jbranch = jg0 * jr0
    derivative = jbranch * v_dir
    eta = sp.symbols("eta", real=True)
    path = gated(core_norm(x0 + eta * v_dir, eps0))
    direct = path.diff(eta).subs(eta, 0)
    matrix_check("D-independent-directional", direct, derivative)
    residual_path = x0 + eta * v_dir + path
    matrix_check("D-residual-directional", residual_path.diff(eta).subs(eta, 0),
                 derivative + v_dir)
    errors = []
    for delta in [Q(1, 100), Q(1, 10000), Q(1, 1000000)]:
        plus = gated(core_norm(x0 + delta * v_dir, eps0)).evalf(60)
        minus = gated(core_norm(x0 - delta * v_dir, eps0)).evalf(60)
        estimate = (plus - minus) / (2 * delta)
        error = max(abs(entry) for entry in estimate - derivative.evalf(60))
        check(f"D-central-error-{delta}", error < 2 * delta**2)
        errors.append({"step": str(delta), "maximum_absolute_error": str(error)})
    check("D-central-convergence", all(sp.Float(errors[i+1]["maximum_absolute_error"])
          < sp.Float(errors[i]["maximum_absolute_error"]) for i in range(2)))

    # D5: independent gate logits for the fixed-head partial derivatives.
    u = sp.Matrix(sp.symbols("u1:3", real=True))
    gate = u.applyfunc(sigmoid)
    projection = sp.Matrix([[1, 1], [0, 1]])
    intermediate = gate.multiply_elementwise(o_star)
    before, after = projection * intermediate, gate.multiply_elementwise(projection * o_star)
    u_fixture = {u[0]: sp.log(3), u[1]: -sp.log(3)}
    matrix_check("D5-projection-difference", (before - after).subs(u_fixture),
                 sp.Matrix([-Q(5, 12), 0]))
    for name, output, expected in (
        ("intermediate", intermediate, sp.diag(Q(1, 8), Q(5, 32))),
        ("before-projection", before, sp.Matrix([[Q(1, 8), Q(5, 32)], [0, Q(5, 32)]])),
        ("after-projection", after, sp.diag(Q(9, 32), Q(5, 32))),
    ):
        matrix_check(f"D5-{name}-logit-J", output.jacobian(u).subs(u_fixture), expected)

    # Let the same query perturb the head and the logits to expose both paths.
    x = sp.Matrix(sp.symbols("q1:3", real=True))
    head = sp.Matrix([x[0] + x[1], 2 * x[1]])
    g = x.applyfunc(sigmoid)
    gate_j = g.jacobian(x)
    head_j = head.jacobian(x)
    matrix_check("D5-before-projection-query-J",
                 (projection * g.multiply_elementwise(head)).jacobian(x),
                 projection * (sp.diag(*head) * gate_j + sp.diag(*g) * head_j))
    matrix_check("D5-after-projection-query-J",
                 g.multiply_elementwise(projection * head).jacobian(x),
                 sp.diag(*(projection * head)) * gate_j + sp.diag(*g) * projection * head_j)

    # E: scalar limits and explicit piecewise corner response.
    t = sp.symbols("t", real=True)
    limits = {}
    for name, expression, expected in [
        ("sigmoid", sigmoid(t), [0, 1, 0, 0]),
        ("tanh", sp.tanh(t), [-1, 1, 0, 0]),
        ("silu", silu(t), [0, sp.oo, 0, 1]),
    ]:
        actual = [sp.limit(expression, t, -sp.oo), sp.limit(expression, t, sp.oo),
                  sp.limit(sp.diff(expression, t), t, -sp.oo),
                  sp.limit(sp.diff(expression, t), t, sp.oo)]
        check(f"E-{name}-four-tail-limits", actual == expected)
        limits[name] = list(map(str, actual))
    equal("E-SiLU-derivative", sp.diff(silu(t), t),
          sigmoid(t) + t*sigmoid(t)*(1-sigmoid(t)))
    k = sp.symbols("k", positive=True)
    equal("E-softclip-derivative", sp.diff(k*sp.tanh(t/k), t),
          1-sp.tanh(t/k)**2)
    # Difference quotients at corners use the two neighboring smooth branches.
    # This avoids accepting a Piecewise corner convention as an ordinary derivative.
    step = sp.symbols("h", positive=True)
    equal("E-clip-upper-left-slope", (1-(1-step))/step, 1)
    equal("E-clip-upper-right-slope", (sp.S.One-1)/step, 0)
    equal("E-clip-lower-left-slope", (-sp.S.One+1)/step, 0)
    equal("E-clip-lower-right-slope", ((-1+step)+1)/step, 1)
    m_coord = sp.symbols("M", positive=True)
    xm = sp.Matrix([m_coord, 1, 0, 0])
    rm = core_norm(xm)
    matrix_check("E-growing-coordinate-limit", rm.applyfunc(
        lambda entry: sp.limit(entry, m_coord, sp.oo)), sp.Matrix([2, 0, 0, 0]))
    equal("E-coordinate-ratio", rm[0]/rm[1], m_coord)
    # The following chi expression uses the explicitly stated domain M >= 1.
    chi_m = 2*m_coord/sp.sqrt(m_coord**2+1)
    equal("E-chi-limit", sp.limit(chi_m, m_coord, sp.oo), 2)
    independent_b = sp.symbols("b", real=True)
    check("E-bounded-factor-unbounded-product",
          sp.limit(sigmoid(0)*independent_b, independent_b, sp.oo) == sp.oo)

    # F: diagonal DyT derivative and exact placement witnesses.
    alpha = sp.symbols("alpha", real=True)
    gamma = sp.Matrix(sp.symbols("g1:5", real=True))
    beta = sp.Matrix(sp.symbols("b1:5", real=True))
    dyt = gamma.multiply_elementwise(xs.applyfunc(lambda t: sp.tanh(alpha*t))) + beta
    dyt_hand = sp.diag(*[gamma[i]*alpha*(1-sp.tanh(alpha*xs[i])**2) for i in range(4)])
    matrix_check("F-DyT-diagonal-J", dyt.jacobian(xs), dyt_hand)
    check("F-DyT-common-shift-counterexample", sp.tanh(1).is_positive)
    matrix_check("F-positive-scalar-cancellation", core_norm(x0/2), core_norm(x0))
    matrix_check("F-positive-eps-scalar-law", core_norm(x0/2, eps0), core_norm(x0, 4*eps0))
    check("F-after-scalar-gate-differs", core_norm(x0)[0] != core_norm(x0)[0]/2)
    diagonal = sp.diag(1, Q(1, 2), 1, Q(1, 2))
    ax = diagonal * x0
    equal("F-diagonal-gate-mean-square", ax.dot(ax)/4, Q(269, 576))
    matrix_check("F-diagonal-normalized-output", core_norm(ax),
                 sp.Matrix([16, 10, 24, -12])/sp.sqrt(269))
    equal("F-diagonal-ratio", core_norm(ax)[0]/core_norm(ax)[1], Q(8, 5))
    # Since a0>0, p1>8/19 and p2<-10/19. Three coordinates already force ||y||>2.
    lower_square = (Q(2, 3)+Q(8, 19))**2 + (1+Q(8, 19))**2 + (1+Q(10, 19))**2
    check("F-pre-norm-lower-bound", lower_square > 4)
    post = core_norm(x0 + gated(x0), eps0)
    check("F-post-value-range", post.dot(post).evalf(30) < 4)
    check("F-pre-post-differ", any(abs(entry.evalf(30)) > Q(1, 10)
          for entry in y0-post))

    # Execute the published starter separately and compare its essential definitions.
    starter = runpy.run_path(str(Path(__file__).with_name("homework_starter.py")))
    for name, expected in [("x0", x0), ("Wg", Wg), ("Wu", Wu), ("Wd", Wd)]:
        matrix_check(f"starter-{name}", starter[name], expected)
    matrix_check("starter-forward", starter["gated"](starter["core_norm"](x0, eps0)), b0)
    rejects("starter-excluded-zero", starter["core_norm"], sp.zeros(4, 1))
    rejects("starter-excluded-constant", starter["core_norm"], ones, center=True)

    return {
        "python": platform.python_version(), "sympy": sp.__version__,
        "checks_passed": len(checks), "checks": checks,
        "moments": {"mean": str(mu), "mean_square": str(q0), "variance": str(variance0)},
        "ranks": rank_table,
        "symmetry_outputs": {
            "R_shifted_by_7": list(map(str, shifted)),
            "R_scaled_by_3_fixed_eps": list(map(str, core_norm(3*x0, eps0))),
            "L_shifted_by_7": list(map(str, core_norm(x0+7*ones, eps0, True))),
            "R_constant_7_positive_eps": list(map(str, core_norm(7*ones, eps0))),
            "L_constant_7_positive_eps": ["0"] * 4,
            "R_shift_first_coordinate_squared_difference": str(sp.simplify(
                shifted[0]**2-core_norm(x0, eps0)[0]**2)),
            "excluded_zero_eps": ["R(0)", "L(7*ones)"],
        },
        "exact_forward": {name: list(map(str, value)) for name, value in
                          [("x", x0), ("h_norm", h_norm0), ("u_g", u_g0), ("u_v", u_v0),
                           ("p", p0), ("b", b0), ("y", y0)]},
        "rounded_branch": list(map(str, b0.evalf(13))),
        "rounded_residual": list(map(str, y0.evalf(13))),
        "directional_branch": list(map(str, derivative.evalf(13))),
        "directional_residual": list(map(str, (derivative + v_dir).evalf(13))),
        "central_differences": errors, "scalar_limits": limits,
        "post_normalized_output": list(map(str, post.evalf(13))),
        "evidence_boundary": "Exact dimension-four identities, constructed-input checks and 60-digit numerical diagnostics; no model or hardware experiment.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write the compact verification record.")
    args = parser.parse_args()
    result = verify()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Passed {result['checks_passed']} checks; Python {result['python']}, SymPy {result['sympy']}.")
    print("Ranks:", result["ranks"])
    print("Branch directional response:", result["directional_branch"])
    for row in result["central_differences"]:
        print("Central difference", row["step"], "max error", row["maximum_absolute_error"])
