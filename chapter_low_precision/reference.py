"""Exact reference solution to the low-precision A–F homework.

Written proofs and interpretations are in doc/ch16/instructor_solutions.md.
Run from the repository root with Python 3 and SymPy.
"""
import sys
import sympy as sp

from fixture_checks import UniformQuantizer, rational, round_even, softmax

R, M = sp.Rational, sp.Matrix
X_INPUT = M([R(2, 3), R(5, 6)])
WEIGHT = M([[1, R(-1, 2)], [R(1, 3), R(2, 3)]])
VALUE = M([[1, 0], [0, 1], [1, 1]])
TARGET = M([0, 1])
QUANT = UniformQuantizer(R(1, 4))


def check(actual, expected):
    """Exact identity, never approximate equality."""
    difference = actual - expected
    terms = list(difference) if isinstance(difference, sp.MatrixBase) else [difference]
    assert all(sp.simplify(term) == 0 for term in terms), (actual, expected)


def squared_norm(value):
    return sp.simplify(sum(entry**2 for entry in value))


def decode(code, quant=QUANT):
    code = rational(code)
    if code.is_Integer is not True or not quant.qmin <= code <= quant.qmax:
        raise ValueError("Decode a legal integer code only.")
    return quant.delta * (code - quant.zero)


def max_quantizer(value):
    magnitude = max(abs(rational(entry)) for entry in value)
    return UniformQuantizer(magnitude / 7 if magnitude else 1)


def clipped_ste_mask(value, quant=QUANT):
    lower = quant.delta * (quant.qmin - quant.zero)
    upper = quant.delta * (quant.qmax - quant.zero)
    return value.applyfunc(lambda entry: sp.Integer(1 if lower < entry < upper else 0))


def scale_surrogate(value, quant=QUANT):
    """Declared LSQ-style rule, specialized to zero point zero."""
    if quant.zero != 0:
        raise ValueError("This learned-scale example requires zero point zero.")
    normalized = rational(value) / quant.delta
    if normalized <= quant.qmin:
        return sp.Integer(quant.qmin)
    if normalized >= quant.qmax:
        return sp.Integer(quant.qmax)
    return round_even(normalized) - normalized


def signed_width(bound):
    if type(bound) is not int or bound < 0:
        raise ValueError("Use a nonnegative integer magnitude bound.")
    width = 1
    while bound > 2**(width - 1) - 1:
        width += 1
    return width


def part_a():
    values = [R(-2), R(-7, 8), R(-5, 8), R(0), R(5, 8), R(7, 8), R(2)]
    codes = M([QUANT.encode(value) for value in values])
    recon = M([decode(code) for code in codes])
    check(codes, M([-7, -4, -2, 0, 2, 4, 7]))
    check(recon, M([R(-7, 4), -1, R(-1, 2), 0, R(1, 2), 1, R(7, 4)]))
    affine = UniformQuantizer(R(1, 2), 0, 7, 3)
    check(affine.encode(R(1, 4)), 3)
    check(round_even(R(1, 4) / affine.delta + affine.zero), 4)
    zero_quant = max_quantizer(M.zeros(2, 2))
    check(zero_quant.delta, 1)
    check(zero_quant.matrix(M.zeros(2, 2)), M.zeros(2, 2))
    return {"codes": codes, "reconstructions": recon, "zero_scale": zero_quant.delta}


def part_b():
    wq, xq = QUANT.matrix(WEIGHT), QUANT.matrix(X_INPUT)
    y = WEIGHT * X_INPUT
    weight_only, activation_only, joint = wq * X_INPUT, WEIGHT * xq, wq * xq
    ew, ex = wq - WEIGHT, xq - X_INPUT
    terms = (WEIGHT * ex, ew * X_INPUT, ew * ex)
    check(y, M([R(1, 4), R(7, 9)]))
    check(weight_only, M([R(1, 4), R(19, 24)]))
    check(activation_only, M([R(3, 8), R(3, 4)]))
    check(joint, activation_only)
    for actual, expected in zip(terms, (M([R(1, 8), R(-1, 36)]),
                                      M([0, R(1, 72)]), M([0, R(-1, 72)]))):
        check(actual, expected)
    check(sum(terms, M.zeros(2, 1)), joint - y)
    codes_w, codes_x = WEIGHT.applyfunc(QUANT.encode), X_INPUT.applyfunc(QUANT.encode)
    accumulator = codes_w * codes_x
    check(accumulator, M([6, 12]))
    check(QUANT.delta**2 * accumulator, joint)
    margin = y[1] - y[0]
    error_inf = max(abs(entry) for entry in joint - y)
    check(margin, R(19, 36))
    check(error_inf, R(1, 8))
    assert error_inf < margin / 2 and joint[1] > joint[0]
    return {"reference": y, "weight_only": weight_only, "activation_only": activation_only,
            "joint": joint, "terms": terms, "accumulator": accumulator,
            "margin": margin, "error_inf": error_inf}


def part_c():
    u = M([R(1, 2), 1, 8])
    shared = max_quantizer(u).matrix(u)
    grouped = M([UniformQuantizer(R(1, 7))(u[0]),
                 UniformQuantizer(R(1, 7))(u[1]), UniformQuantizer(R(8, 7))(u[2])])
    check(shared, M([0, R(8, 7), 8]))
    check(grouped, M([R(4, 7), 1, 8]))
    shared_error, grouped_error = squared_norm(shared-u), squared_norm(grouped-u)
    check(shared_error, R(53, 196))
    check(grouped_error, R(1, 196))
    wo, xo, diagonal = M([[1, R(1, 8)]]), M([1, 8]), sp.diag(1, 8)
    wt, xt = wo * diagonal, diagonal.inv() * xo
    check(wo * xo, wt * xt)
    original_q = (max_quantizer(wo).matrix(wo) * max_quantizer(xo).matrix(xo))[0]
    transformed_q = (max_quantizer(wt).matrix(wt) * max_quantizer(xt).matrix(xt))[0]
    check(original_q, R(16, 7))
    check(transformed_q, 2)
    calibration = M.hstack(X_INPUT, M([1, 0]), M([0, 1]))
    moment = calibration * calibration.T / 3
    ew = QUANT.matrix(WEIGHT) - WEIGHT
    loss_direct = squared_norm(ew * calibration) / 6
    loss_trace = sp.trace(ew * moment * ew.T) / 2
    check(moment, M([[R(13, 27), R(5, 27)], [R(5, 27), R(61, 108)]]))
    check(loss_direct, loss_trace)
    check(loss_trace, R(73, 31104))
    h1, h2 = sp.symbols("h1 h2", real=True)
    h = M([h1, h2])
    check(sp.hessian((h.T * moment * h)[0] / 2, (h1, h2)), moment)
    anisotropic = M([[1, 1], [10, -10]])
    check(anisotropic * anisotropic.T / 2, sp.diag(1, 100))
    return {"shared": shared, "grouped": grouped, "errors": (shared_error, grouped_error),
            "scale_counts": (1, 2), "rescaled_outputs": (original_q, transformed_q),
            "moment": moment, "reconstruction_loss": loss_trace}


def calibration_comparison(a=R(1, 100), b=R(10)):
    """Optional two-column comparison; independent of Part C's three inputs.

    The two reconstructions are stipulated directly, not quantizer outputs.
    Positive exact inputs also permit the exercise's symbolic a, b.
    """
    reference_weight = sp.diag(10, 1)
    comparison_inputs = sp.diag(a, b)
    comparison_count = comparison_inputs.cols
    comparison_moment = comparison_inputs * comparison_inputs.T / comparison_count
    candidates = (sp.diag(10, 0), sp.diag(0, 1))
    records = []
    for candidate in candidates:
        residual = candidate - reference_weight
        output_error = residual * comparison_inputs
        direct_loss = squared_norm(output_error) / (2 * comparison_count)
        trace_loss = sp.trace(residual * comparison_moment * residual.T) / 2
        check(direct_loss, trace_loss)
        records.append({"residual": residual, "output_error": output_error,
                        "weight_norm": sp.sqrt(squared_norm(residual)),
                        "output_norm": sp.sqrt(squared_norm(output_error)),
                        "loss": direct_loss, "trace_loss": trace_loss})
    return {"inputs": comparison_inputs, "count": comparison_count,
            "moment": comparison_moment, "candidates": records}


def part_d():
    logits = M([0, sp.log(2), sp.log(3)])
    changed_logits = M([0, sp.log(2), sp.log(4)])  # analytic score probe, not Q(logits)
    a, changed_a = softmax(logits), softmax(changed_logits)
    z, changed_z = VALUE.T * a, VALUE.T * changed_a
    check(a, M([R(1, 6), R(1, 3), R(1, 2)]))
    check(z, X_INPUT)
    check(changed_a, M([R(1, 7), R(2, 7), R(4, 7)]))
    check(changed_z, M([R(5, 7), R(6, 7)]))
    check(changed_z - z, M([R(1, 21), R(1, 42)]))
    jacobian = sp.diag(*a) - a*a.T
    check(jacobian * M.ones(3, 1), M.zeros(3, 1))
    check(jacobian, jacobian.T)
    score_bound_squared = (changed_logits-logits).dot(changed_logits-logits) / 4
    assert bool(sp.simplify(score_bound_squared - squared_norm(changed_a-a)) > 0)
    return {"probabilities": a, "output": z, "analytic_probabilities": changed_a,
            "analytic_output": changed_z, "jacobian": jacobian}


def part_e():
    wq, xq = QUANT.matrix(WEIGHT), QUANT.matrix(X_INPUT)
    error = wq * xq - TARGET
    loss = squared_norm(error) / 2
    weight_gradient = (error * xq.T).multiply_elementwise(clipped_ste_mask(WEIGHT))
    input_gradient = (wq.T * error).multiply_elementwise(clipped_ste_mask(X_INPUT))
    check(loss, R(13, 128))
    check(weight_gradient, M([[R(9, 32), R(9, 32)], [R(-3, 16), R(-3, 16)]]))
    check(input_gradient, M([R(5, 16), R(-3, 8)]))
    master_next = WEIGHT - R(1, 4) * weight_gradient
    check(master_next, M([[R(119, 128), R(-73, 128)], [R(73, 192), R(137, 192)]]))
    reconstructed_next = QUANT.matrix(master_next)
    output_next = reconstructed_next * xq
    loss_next = squared_norm(output_next - TARGET) / 2
    check(reconstructed_next, M([[1, R(-1, 2)], [R(1, 2), R(3, 4)]]))
    check(output_next, M([R(3, 8), R(15, 16)]))
    check(loss_next, R(37, 512))
    assert loss_next < loss
    changed_entries = [(i, j) for i in range(2) for j in range(2)
                       if reconstructed_next[i, j] != wq[i, j]]
    assert changed_entries == [(1, 0)]
    w = R(1, 3)
    scalar_gradient = QUANT(w) - 1
    master_scalar_next = w - R(1, 100) * scalar_gradient
    check(master_scalar_next, R(409, 1200))
    assert R(1, 8) < master_scalar_next < R(3, 8)
    check(QUANT(master_scalar_next), QUANT(w))
    check(scalar_gradient, R(-3, 4))
    check(scale_surrogate(w), R(-1, 3))
    # On 2/9 < delta < 2/3 the actual code is 1: ordinary local derivative is 1.
    step = sp.symbols("step", positive=True)
    ordinary_scale_derivative = sp.diff(step * 1, step)
    check(ordinary_scale_derivative, 1)
    check(squared_norm(WEIGHT*X_INPUT-TARGET)/2, R(145, 2592))
    return {"weight_surrogate": weight_gradient, "input_surrogate": input_gradient,
            "master_next": master_next, "output_next": output_next,
            "losses": (loss, loss_next), "changed_entries": changed_entries,
            "scalar_master_next": master_scalar_next,
            "ordinary_input_derivative": sp.Integer(0), "scalar_surrogate": scalar_gradient,
            "ordinary_scale_derivative": ordinary_scale_derivative,
            "scale_surrogate": scale_surrogate(w)}


def part_f():
    bits = 1024*4 + 8*16
    assert bits % 8 == 0
    byte_count = bits // 8
    bound = 6*7*7
    width = signed_width(bound)
    assert byte_count == 528 and bound == 294 and width == 10
    ratio = R(1024*16, bits)
    check(ratio, R(128, 33))
    ledger = {
        "payload": "15 active narrow-symmetric integer codes; unused pattern reserved",
        "scale": "exact rational in homework; 16-bit stored metadata in storage probe",
        "zero": "implicit zero, except explicit affine Part A probe",
        "groups": "per-operand tensor in readout; one/two groups in outlier probe",
        "rounding": "nearest-even; round before adding zero point",
        "clipping": "integer bounds -7,7 in readout",
        "multiplication": "exact integer/rational calculation, no hardware simulation",
        "accumulation": "exact toy sum; conservative signed width bound only",
        "output_conversion": "rational reconstruction, no bias/output requantization",
        "masters": "exact rational W updated; production dtype unspecified",
        "gradients": "explicit rational surrogate; production dtype unspecified",
        "optimizer_state": "one SGD step, no moments; production state unspecified",
        "evaluation": "fixed constructed inputs, no held-out model evaluation",
    }
    return {"bytes": byte_count, "ratio": ratio, "bound": bound, "width": width,
            "ledger": ledger}


def run():
    print("Python:", sys.version.split()[0])
    print("SymPy:", sp.__version__)
    results = {name: function() for name, function in zip(
        "ABCDEF", (part_a, part_b, part_c, part_d, part_e, part_f))}
    print("PASS: exact reference solution for A–F.")
    print("Joint output:", results["B"]["joint"].T)
    print("QAT losses:", results["E"]["losses"])
    print("Storage bytes / accumulator width:", results["F"]["bytes"], results["F"]["width"])
    comparison = calibration_comparison()
    print("Optional reconstruction comparison losses:",
          [row["loss"] for row in comparison["candidates"]])
    print("Evidence: rational reconstruction only; no hardware/model-quality experiment.")
    return results


if __name__ == "__main__":
    run()
