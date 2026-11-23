"""Check the four-part core homework; optionally run historical extensions.

Usage from repository root:
  python3 code/chapter_differentiation/ch03_instructor_checks.py --output-dir /tmp/ch03

Assertions establish identities for this explicit model or properties of the
specified fixtures; they do not prove arbitrary-dimensional graph correctness.
"""

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import sympy as sp

from ch03_instructor_solution import (
    central_difference, directions, fixture_values, historical_fixture_values,
    rvec, scalar_loss, shared_model, symbolic_model,
)


def require_zero(expressions, transform, name):
    residuals = [transform(expression) for expression in expressions]
    unresolved = [(i, residual) for i, residual in enumerate(residuals) if residual != 0]
    if unresolved:
        raise AssertionError(f"{name}: unresolved residuals (not proof of inequality): {unresolved}")
    return len(residuals)


def check_symbolic(model):
    counts = {}
    counts["gradient"] = require_zero(model["gradient"] - model["assembled"], sp.cancel,
                                      "independent scalar versus matrix gradients")
    counts["probability_rows"] = require_zero(
        [sum(model["P"][i, j] for j in range(3)) - 1 for i in range(2)],
        sp.cancel, "probability row sums")
    counts["gradient_rows"] = require_zero(
        [sum(model["G"][i, j] for j in range(3)) for i in range(2)],
        sp.cancel, "logit gradient row sums")

    zs = sp.symbols("z0:3", real=True)
    p = sp.Matrix([sp.exp(z) / sum(sp.exp(q) for q in zs) for z in zs])
    softmax_rule = sp.diag(*p) - p * p.T
    counts["softmax_J"] = require_zero(p.jacobian(zs) - softmax_rule, sp.cancel,
                                       "three-class softmax Jacobian")
    ell = -sp.log(p[2])
    counts["row_loss"] = require_zero(
        [sp.diff(ell, zs[j]) - (p[j] - int(j == 2)) for j in range(3)],
        sp.cancel, "one-row cross-entropy gradient")
    return counts


def check_jacobian(model):
    X, W, Z = model["X"], model["W"], model["Z"]
    counts = {}
    # Matrix rules are written before computing products of the explicit Jacobian.
    vs = sp.symbols("v0:13", real=True)
    v = sp.Matrix(vs)
    dX, dW, db = sp.Matrix(2, 2, vs[:4]), sp.Matrix(2, 3, vs[4:10]), sp.Matrix(vs[10:])
    tangent = dX * W + X * dW + sp.ones(2, 1) * db.T
    us = sp.symbols("u0:6", real=True)
    U = sp.Matrix(2, 3, us)
    pullback = rvec(U * W.T).col_join(rvec(X.T * U)).col_join(U.T * sp.ones(2, 1))
    J = rvec(Z).jacobian(model["theta"])
    assert J.shape == (6, 13)
    counts["JVP"] = require_zero(J * v - rvec(tangent), sp.expand, "matrix JVP")
    counts["VJP"] = require_zero(J.T * rvec(U) - pullback, sp.expand, "matrix VJP")
    counts["adjoint"] = require_zero(
        [(rvec(U).T * rvec(tangent))[0] - (pullback.T * v)[0]],
        sp.expand, "independent matrix adjoint identity")

    return counts


def check_fixtures(model):
    R = sp.Rational
    expected = {
        "zero": dict(Z=sp.zeros(2, 3), P=sp.ones(2, 3) / 3, L=sp.log(3),
                     G=sp.Matrix([[-R(1, 3), R(1, 6), R(1, 6)],
                                  [R(1, 6), R(1, 6), -R(1, 3)]]),
                     barX=sp.zeros(2, 2),
                     barW=sp.Matrix([[-R(1, 3), R(1, 6), R(1, 6)],
                                     [-R(1, 6), R(1, 3), -R(1, 6)]]),
                     barb=sp.Matrix([-R(1, 6), R(1, 3), -R(1, 6)])),
        "nonzero": dict(Z=sp.Matrix([[sp.log(2), 0, 0], [0, sp.log(3), 0]]),
                        P=sp.Matrix([[R(1, 2), R(1, 4), R(1, 4)],
                                     [R(1, 5), R(3, 5), R(1, 5)]]),
                        L=sp.log(10) / 2,
                        G=sp.Matrix([[-R(1, 4), R(1, 8), R(1, 8)],
                                     [R(1, 10), R(3, 10), -R(2, 5)]]),
                        barX=sp.Matrix([[-sp.log(2)/4-sp.log(3)/8, sp.log(3)/8],
                                        [sp.log(2)/10-3*sp.log(3)/10, 3*sp.log(3)/10]]),
                        barW=sp.Matrix([[-R(1, 4), R(1, 8), R(1, 8)],
                                        [-R(3, 20), R(17, 40), -R(11, 40)]]),
                        barb=sp.Matrix([-R(3, 20), R(17, 40), -R(11, 40)])),
    }
    outputs = {}
    for name, fixture in expected.items():
        point = dict(zip(model["theta"], fixture_values(name == "zero")))
        for key, value in fixture.items():
            residual = model[key].subs(point) - value
            entries = list(residual) if isinstance(residual, sp.MatrixBase) else [residual]
            require_zero(entries, lambda x: sp.cancel(sp.logcombine(sp.expand(x))), f"{name} {key}")
        # Check fixture answers directly against scalar differentiation too.
        wanted = rvec(fixture["barX"]).col_join(rvec(fixture["barW"])).col_join(fixture["barb"])
        require_zero(model["gradient"].subs(point) - wanted, sp.cancel,
                     f"{name} independent scalar derivatives")
        require_zero(fixture["G"] * sp.ones(3, 1), sp.cancel, f"{name} zero row sums")
        outputs[name] = {key: str(value) for key, value in fixture.items()}
    return outputs


def check_shared():
    m = shared_model()
    counts = {}
    counts["shared_full"] = require_zero(m["full_gradient"] - m["r"] * (m["a"]-m["b"]).T,
                                         sp.expand, "ordinary shared derivative")
    counts["shared_stopped"] = require_zero(m["stopped"] - m["r"] * m["a"].T,
                                            sp.expand, "differentiate then substitute target")
    restored = m["frozen"].subs(m["target"], simultaneous=True)
    restored_gradient = sp.Matrix(2, 2, [sp.diff(restored, x) for x in m["entries"]])
    counts["early_substitution"] = require_zero(restored_gradient - m["full_gradient"],
                                               sp.expand, "substitute before differentiation")
    point = dict(zip(m["entries"], [1, 2, 0, 1]))
    point.update(zip(m["a"], [1, 0]))
    point.update(zip(m["b"], [1, 1]))
    assert m["L"].subs(point) == sp.Rational(5, 2)
    assert m["full_gradient"].subs(point) == sp.Matrix([[0, 2], [0, 1]])
    assert m["stopped"].subs(point) == sp.Matrix([[-2, 0], [-1, 0]])

    # Independent exact directional checks perturb both branches or freeze q ONCE.
    # Quadratics make central differences exact for these rational nonzero steps.
    target_at_reference = (m["A"] * m["b"]).subs(point)
    frozen_at_reference = m["frozen"].subs(dict(zip(m["q"], target_at_reference)))
    directions_exact = [sp.Matrix([[1, 0], [0, 0]]), sp.Matrix([[0, 1], [1, -1]])]
    for direction in directions_exact:
        h = sp.Rational(1, 10)
        plus = {**point, **{x: point[x]+h*v for x, v in zip(m["entries"], direction)}}
        minus = {**point, **{x: point[x]-h*v for x, v in zip(m["entries"], direction)}}
        for label, loss, gradient in (("recomputed", m["L"], m["full_gradient"]),
                                       ("frozen", frozen_at_reference, m["stopped"])):
            estimate = (loss.subs(plus)-loss.subs(minus))/(2*h)
            pairing = sum(g*v for g, v in zip(gradient.subs(point), direction))
            require_zero([estimate-pairing], sp.expand, f"{label} exact directional check")
    # The first direction separates full and stopped sensitivities: 0 versus -2.
    assert m["full_gradient"].subs(point)[0, 0] == 0
    assert m["stopped"].subs(point)[0, 0] == -2
    return counts


def numerical_protocol(model):
    """Optional historical rational-fixture experiment, independent of core fixtures."""
    import mpmath as mp

    rows, bugs, reference_pairings = [], [], {}
    # References originate from scalar sp.diff, never from the candidate rules.
    gradient_float = sp.lambdify(model["theta"], list(model["gradient"]), "math")
    gradient_mp = sp.lambdify(model["theta"], list(model["gradient"]), "mpmath")
    with mp.workdps(80):
        for mode, arithmetic, gradient_function in (
            ("float", math, gradient_float), ("mp80", mp, gradient_mp),
        ):
            exact = historical_fixture_values()
            values = ([float(x) for x in exact] if mode == "float" else
                      [mp.mpf(int(x.p)) / int(x.q) for x in exact])
            gradient = gradient_function(*values)
            ds = directions(arithmetic)
            assert all(abs(sum(x * x for x in v) - 1) < arithmetic.sqrt(1) * 1e-14 for v in ds)
            reference_pairings[mode] = []
            for number, direction in enumerate(ds, start=1):
                reference = sum(g * v for g, v in zip(gradient, direction))
                reference_pairings[mode].append(str(reference))
                for k in range(1, 13):
                    step = 10.0 ** (-k) if mode == "float" else mp.mpf(10) ** (-k)
                    estimate = central_difference(values, direction, step, arithmetic)
                    error = abs(estimate - reference)
                    assert arithmetic.isfinite(error)
                    rows.append(dict(mode=mode, direction=number, k=k, h=str(step),
                                     reference=str(reference), central_difference=str(estimate),
                                     absolute_error=str(error)))
                    if k == 6:
                        assert error < (1e-8 if mode == "float" else mp.mpf("1e-10"))

            # Separate plausible corruptions, each detected by the middle-bias direction.
            step = 1e-5 if mode == "float" else mp.mpf("1e-5")
            estimate = central_difference(values, ds[2], step, arithmetic)
            missing_mean = [2 * g for g in gradient]
            zero_bias = [*gradient[:10], 0, 0, 0]
            for name, candidate in (("missing_mean", missing_mean), ("zero_bias", zero_bias)):
                candidate_pairing = sum(g * v for g, v in zip(candidate, ds[2]))
                error = abs(estimate - candidate_pairing)
                assert error > 1e-3
                bugs.append(dict(mode=mode, bug=name, direction=3, absolute_error=str(error)))

            shifted = scalar_loss(values, arithmetic, common_shift=1000)
            original = scalar_loss(values, arithmetic)
            assert abs(shifted - original) < (1e-12 if mode == "float" else mp.mpf("1e-70"))
        assert len(rows) == 72
        assert len({(row["mode"], row["direction"], row["k"]) for row in rows}) == 72
        # This observation is about ordinary Python's binary64 arithmetic on this host.
        assert (1.0 + 2.0 ** -54) - 1.0 == 0.0
        return rows, bugs, reference_pairings


def run(output_dir=None, extensions=False):
    model = symbolic_model()
    counts = check_symbolic(model)
    fixture = check_fixtures(model)
    counts.update(check_shared())
    summary = dict(versions=dict(Python=platform.python_version(), SymPy=sp.__version__),
                   symbolic_residual_counts=counts, fixtures=fixture,
                   frozen_target_directional_cases=4)
    rows = []
    if extensions:
        import mpmath as mp
        counts.update(check_jacobian(model))
        rows, bugs, pairings = numerical_protocol(model)
        summary["optional_historical_extensions"] = dict(
            fixture=list(map(str, historical_fixture_values())), mpmath=mp.__version__,
            numerical_cases=len(rows), detected_bugs=bugs, reference_pairings=pairings)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            with (output_dir / "directional_sweep.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--extensions", action="store_true",
                        help="also run the full Jacobian and historical 72-case precision sweep")
    args = parser.parse_args()
    run(args.output_dir, args.extensions)
