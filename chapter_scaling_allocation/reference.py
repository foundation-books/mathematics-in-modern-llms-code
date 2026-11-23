"""Exact constructed allocation fixtures; these are not training measurements."""

import sys

import sympy as sp


n, d, c = sp.symbols("n d c", positive=True)
loss = 1 + 4 / n + 9 / d


def fit_fixture(perturb=False):
    """Fixed exponents, raw-loss least squares, nine fully specified rows."""
    grid = [(sp.Integer(ni), sp.Integer(di))
            for ni in (4, 8, 16) for di in (9, 18, 36)]
    design = sp.Matrix([[1, 1 / ni, 1 / di] for ni, di in grid])
    responses = sp.Matrix([loss.subs({n: ni, d: di}) for ni, di in grid])
    if perturb:
        responses[grid.index((sp.Integer(4), sp.Integer(18)))] += sp.Rational(1, 50)
    coefficients = (design.T * design).inv() * design.T * responses
    return grid, design, responses, coefficients


def reciprocal_optimum(compute, model_coefficient=4, data_coefficient=9,
                       token_cap=None):
    """Minimize A/n+B/d at nd=compute, with an optional positive d cap."""
    compute, a, b = map(sp.sympify, (compute, model_coefficient, data_coefficient))
    if any(v.is_positive is not True for v in (compute, a, b)):
        raise ValueError("compute and coefficients must be positive")
    model_size = sp.sqrt(a * compute / b)
    if token_cap is not None:
        cap = sp.sympify(token_cap)
        if cap.is_positive is not True:
            raise ValueError("token cap must be positive")
        model_size = sp.Max(model_size, compute / cap)
    return model_size, sp.simplify(compute / model_size)


def effective_exposures(unique, passes, discount):
    """Finite sum with explicit rho=1 behavior and integer pass count."""
    unique, discount = map(sp.sympify, (unique, discount))
    if unique.is_positive is not True or not isinstance(passes, int) or passes < 1:
        raise ValueError("positive unique count and positive integer passes required")
    if discount.is_nonnegative is not True or (1 - discount).is_nonnegative is not True:
        raise ValueError("discount must lie in [0,1]")
    return sp.simplify(unique * sum(discount ** k for k in range(passes)))


def lifetime_optimum(inference_weight):
    """Fixed target loss 2; no data cap is imposed in this problem."""
    weight = sp.sympify(inference_weight)
    if weight.is_nonnegative is not True:
        raise ValueError("inference weight must be nonnegative")
    return 4 + 12 / sp.sqrt(9 + weight), 9 + 3 * sp.sqrt(9 + weight)


def verify_all():
    """Tasks A--F, all fixture anchors, endpoints, and independent identities."""
    grid, design, responses, theta = fit_fixture()
    assert design.rank() == 3 and theta == sp.Matrix([1, 4, 9])
    assert design * theta == responses
    heldout = sp.Matrix([[1, sp.Rational(1, 12), sp.Rational(1, 24)]])
    assert (heldout * theta)[0] == sp.Rational(41, 24)
    confounded = sp.Matrix([[1, 1 / ni, 4 / (9 * ni)]
                            for ni in map(sp.Integer, (4, 8, 16))])
    assert confounded.rank() == 2
    assert confounded * sp.Matrix([0, -sp.Rational(4, 9), 1]) == sp.zeros(3, 1)

    baseline = [(4, 36, sp.Rational(9, 4)), (8, 18, 2),
                (16, 9, sp.Rational(9, 4)), (12, 12, sp.Rational(25, 12))]
    for ni, di, expected in baseline:
        assert ni * di == 144 and loss.subs({n: ni, d: di}) == expected
    assert reciprocal_optimum(144) == (8, 18)
    assert reciprocal_optimum(576) == (16, 36)
    assert loss.subs({n: 16, d: 36}) == sp.Rational(3, 2)
    assert reciprocal_optimum(144, token_cap=12) == (12, 12)
    a, b, alpha, beta = sp.symbols("A B alpha beta", positive=True)
    z = sp.symbols("z", real=True)
    phi = a * sp.exp(-alpha * z) + b * c ** (-beta) * sp.exp(beta * z)
    assert sp.simplify(sp.diff(phi, z, 2) -
                       (alpha ** 2 * a * sp.exp(-alpha * z) +
                        beta ** 2 * b * c ** (-beta) * sp.exp(beta * z))) == 0
    for ap, bp in [(sp.Rational(1, 2), sp.Rational(3, 2)),
                   (sp.Rational(2, 3), sp.Rational(1, 3))]:
        ns = (ap * 4 / (bp * 9)) ** (1 / (ap + bp)) * 144 ** (bp / (ap + bp))
        ds = 144 / ns
        assert sp.simplify(ap * 4 * ns ** (-ap) - bp * 9 * ds ** (-bp)) == 0

    t = sp.symbols("d1", positive=True)
    data_term = 1 / t + 4 / (d - t)
    assert sp.simplify(sp.diff(data_term, t).subs(t, d / 3)) == 0
    assert sp.simplify(data_term.subs(t, d / 3)) == 9 / d
    aggregate = 1 + 4 / n + data_term
    for allocation, expected in [(6, 2), (9, sp.Rational(37, 18)),
                                 (10, sp.Rational(21, 10))]:
        assert aggregate.subs({n: 8, d: 18, t: allocation}) == expected
    assert sp.diff(data_term, t).subs({d: 18, t: 10}) == sp.Rational(21, 400)
    assert effective_exposures(6, 3, sp.Rational(1, 2)) == sp.Rational(21, 2)
    assert effective_exposures(6, 3, 0) == 6
    assert effective_exposures(6, 3, 1) == 18
    rho = sp.symbols("rho", nonnegative=True)
    for m in range(1, 7):
        assert sp.cancel(sum(rho ** k for k in range(m)) - (1 - rho ** m) / (1 - rho)) == 0
    assert sp.Rational(21, 2) <= 18 and 6 / (1 - sp.Rational(1, 2)) == 12
    assert 6 * 100 * 2 + 2000 == 3200
    assert 6 * 100 * 10 + 2000 == 8000

    tau = sp.symbols("tau", nonnegative=True)
    ns, ds = lifetime_optimum(tau)
    quality_data = 9 * n / (n - 4)
    lifetime = n * (quality_data + tau)
    assert sp.simplify(sp.diff(lifetime, n) - (9 + tau - 144 / (n - 4) ** 2)) == 0
    assert sp.simplify(sp.diff(lifetime, n, 2) - 288 / (n - 4) ** 3) == 0
    assert sp.simplify(sp.diff(lifetime, n).subs(n, ns)) == 0
    assert sp.simplify(quality_data.subs(n, ns) - ds) == 0
    assert sp.simplify(loss.subs({n: ns, d: ds}) - 2) == 0
    assert lifetime_optimum(0) == (8, 18) and lifetime_optimum(7) == (7, 21)
    assert 7 * 21 == 147 and 7 * 7 == 49 and 7 * (21 + 7) == 196
    assert 8 * (18 + 7) == 200

    _, _, _, perturbed = fit_fixture(perturb=True)
    assert perturbed == sp.Matrix([sp.Rational(449, 450), sp.Rational(424, 105),
                                   sp.Rational(1572, 175)])
    assert (heldout * perturbed)[0] == sp.Rational(299, 175)
    ns, ds = reciprocal_optimum(144, perturbed[1], perturbed[2])
    regret = sp.simplify(loss.subs({n: ns, d: ds}) - 2)
    assert regret > 0
    ratio = sp.simplify(ns / 8)
    assert sp.simplify(regret - (ratio + 1 / ratio) / 2 + 1) == 0
    r = sp.symbols("r", positive=True)
    f, g = 1 + r ** (-sp.Rational(1, 2)), sp.Rational(4, 3) + sp.Rational(2, 3) / r
    for ri, expected in [(1, 2), (4, sp.Rational(3, 2))]:
        assert f.subs(r, ri) == g.subs(r, ri) == expected
    assert f.subs(r, 16) == sp.Rational(5, 4) and g.subs(r, 16) == sp.Rational(11, 8)
    assert sp.limit(f, r, sp.oo) == 1 and sp.limit(g, r, sp.oo) == sp.Rational(4, 3)
    return perturbed, ns, ds, regret


if __name__ == "__main__":
    print("Python:", sys.version.split()[0], "SymPy:", sp.__version__)
    coefficients, size, tokens, regret = verify_all()
    print("All exact constructed fixture checks passed.")
    print("F1(a) Response-surface sensitivity; refitted coefficients:", tuple(coefficients))
    print("Perturbed allocation:", size, tokens)
    print("Allocation decimals:", sp.N(size, 15), sp.N(tokens, 15))
    print("Model-conditional regret:", regret, sp.N(regret, 25))
