"""Exact planning fixtures for Chapter 16; not a low-precision hardware emulator.

Run with Python 3 and SymPy. Every quantizer input must be exact rational.
The script is an author reference, not the student homework solution template.
"""
from dataclasses import dataclass
import sympy as sp

R = sp.Rational
M = sp.Matrix
CHECKS = []


def rational(value):
    value = sp.sympify(value)
    if value.is_Rational is not True:
        raise ValueError("Use exact integers or SymPy Rational values, not floats.")
    return value


def round_even(value):
    """Nearest integer, with exact ties to the even integer; handles negatives."""
    value = rational(value)
    low = sp.floor(value)
    fraction = value - low
    if fraction < R(1, 2):
        return low
    if fraction > R(1, 2):
        return low + 1
    return low if int(low) % 2 == 0 else low + 1


@dataclass(frozen=True)
class UniformQuantizer:
    delta: object
    qmin: int = -7
    qmax: int = 7
    zero: int = 0

    def __post_init__(self):
        object.__setattr__(self, "delta", rational(self.delta))
        if self.delta <= 0:
            raise ValueError("The scale must be positive.")
        if any(type(v) is not int for v in (self.qmin, self.qmax, self.zero)):
            raise ValueError("Code bounds and zero point must be Python integers.")
        if not self.qmin < self.qmax or not self.qmin <= self.zero <= self.qmax:
            raise ValueError("Require qmin < qmax and an in-range zero point.")

    def encode(self, value):
        # The convention is round first, THEN add the zero point.
        code = round_even(rational(value) / self.delta) + self.zero
        return sp.Integer(min(self.qmax, max(self.qmin, int(code))))

    def __call__(self, value):
        return self.delta * (self.encode(value) - self.zero)

    def matrix(self, value):
        return value.applyfunc(self)


def equal(name, actual, expected):
    difference = actual - expected
    entries = list(difference) if isinstance(difference, sp.MatrixBase) else [difference]
    if any(sp.simplify(entry) != 0 for entry in entries):
        raise AssertionError(f"{name}: {actual!s} != {expected!s}")
    CHECKS.append(name)


def true(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def rejects(name, function):
    try:
        function()
    except ValueError:
        CHECKS.append(name)
    else:
        raise AssertionError(name)


def softmax(logits):
    terms = logits.applyfunc(sp.exp)
    return terms / sum(terms)


def squared_frobenius(matrix):
    return sp.simplify(sum(entry**2 for entry in matrix))


def run():
    CHECKS.clear()
    quant = UniformQuantizer(R(1, 4))
    equal("rounding: positive even tie", round_even(R(5, 2)), 2)
    equal("rounding: positive odd tie", round_even(R(7, 2)), 4)
    equal("rounding: negative even tie", round_even(R(-5, 2)), -2)
    equal("rounding: negative odd tie", round_even(R(-7, 2)), -4)
    equal("rounding: negative non-tie", round_even(R(-7, 3)), -2)
    equal("rounding: exact zero", quant(0), 0)
    equal("rounding: upper clipping", quant(2), R(7, 4))
    equal("rounding: lower clipping", quant(-2), R(-7, 4))
    affine = UniformQuantizer(R(1, 2), 0, 7, 3)
    equal("affine: encoded zero", affine.encode(0), 3)
    equal("affine: decoded zero", affine(0), 0)
    equal("affine: lower endpoint", affine(-3), R(-3, 2))
    equal("affine: upper endpoint", affine(3), 2)
    equal("affine: rounding order at a tie", affine.encode(R(1, 4)), 3)
    rejects("validation: no zero scale", lambda: UniformQuantizer(0))
    rejects("validation: reject float inputs", lambda: quant(0.1))
    rejects("validation: zero point in code range", lambda: UniformQuantizer(1, -7, 7, 8))
    grid = [R(j, 48) for j in range(-84, 85)]
    true("scalar bound on a finite in-range test grid", all(abs(quant(t)-t) <= R(1, 8) for t in grid))
    true("idempotence on a finite test grid", all(quant(quant(t)) == quant(t) for t in grid))

    x = M([R(2, 3), R(5, 6)])
    W = M([[1, R(-1, 2)], [R(1, 3), R(2, 3)]])
    target = M([0, 1])
    xq, Wq = quant.matrix(x), quant.matrix(W)
    y, yq = W*x, Wq*xq
    equal("fixture: full precision output", y, M([R(1, 4), R(7, 9)]))
    equal("fixture: quantized activation", xq, M([R(3, 4), R(3, 4)]))
    equal("fixture: quantized weight", Wq, M([[1, R(-1, 2)], [R(1, 4), R(3, 4)]]))
    equal("fixture: weight-only output", Wq*x, M([R(1, 4), R(19, 24)]))
    equal("fixture: activation-only output", W*xq, M([R(3, 8), R(3, 4)]))
    equal("fixture: joint quantized output", yq, M([R(3, 8), R(3, 4)]))
    Ew, ex = Wq-W, xq-x
    equal("error: activation perturbation", ex, M([R(1, 12), R(-1, 12)]))
    equal("error: propagated activation term", W*ex, M([R(1, 8), R(-1, 36)]))
    equal("error: weight term", Ew*x, M([0, R(1, 72)]))
    equal("error: nonzero cross term", Ew*ex, M([0, R(-1, 72)]))
    equal("error: exact decomposition", yq-y, W*ex+Ew*x+Ew*ex)
    equal("error: final perturbation", yq-y, M([R(1, 8), R(-1, 36)]))
    qw, qx = W.applyfunc(quant.encode), x.applyfunc(quant.encode)
    equal("integer: accumulator values", qw*qx, M([6, 12]))
    equal("integer: decoded dot products", R(1, 16)*(qw*qx), yq)
    equal("margin: reference gap", y[1]-y[0], R(19, 36))
    true("margin: sufficient condition", max(abs(t) for t in yq-y) < (y[1]-y[0])/2)

    logits = M([0, sp.log(2), sp.log(3)])
    values = M([[1, 0], [0, 1], [1, 1]])  # Local compatible realization.
    a = softmax(logits)
    equal("attention: reference probabilities", a, M([R(1, 6), R(1, 3), R(1, 2)]))
    equal("attention: compatible realization", values.T*a, x)
    logits_new = M([0, sp.log(2), sp.log(4)])  # Analytic perturbation, not the 1/4 quantizer.
    a_new = softmax(logits_new)
    equal("attention: perturbed probabilities", a_new, M([R(1, 7), R(2, 7), R(4, 7)]))
    equal("attention: perturbed output", values.T*a_new, M([R(5, 7), R(6, 7)]))
    equal("attention: output perturbation", values.T*(a_new-a), M([R(1, 21), R(1, 42)]))
    J = sp.diag(*a)-a*a.T
    equal("attention: constant-shift nullspace", J*sp.ones(3, 1), sp.zeros(3, 1))
    true("attention: Jacobian row-sum bound at fixture", all(sum(abs(J[i,j]) for j in range(3)) <= R(1, 2) for i in range(3)))

    outliers = M([R(1, 2), 1, 8])
    shared = UniformQuantizer(R(8, 7)).matrix(outliers)
    grouped = M([UniformQuantizer(R(1, 7))(outliers[0]), 1, 8])
    equal("outlier: one-scale reconstruction", shared, M([0, R(8, 7), 8]))
    equal("outlier: grouped reconstruction", grouped, M([R(4, 7), 1, 8]))
    equal("outlier: shared squared error", squared_frobenius(shared-outliers), R(53, 196))
    equal("outlier: grouped squared error", squared_frobenius(grouped-outliers), R(1, 196))
    xo, Wo, D = M([1, 8]), M([[1, R(1, 8)]]), sp.diag(1, 8)
    equal("rescaling: exact identity", Wo*xo, (Wo*D)*(D.inv()*xo))
    equal("rescaling: original quantized output", UniformQuantizer(R(1, 7)).matrix(Wo)*UniformQuantizer(R(8, 7)).matrix(xo), M([R(16, 7)]))
    equal("rescaling: transformed quantized output", UniformQuantizer(R(1, 7)).matrix(Wo*D)*UniformQuantizer(R(1, 7)).matrix(D.inv()*xo), M([2]))

    X = M.hstack(x, M([1, 0]), M([0, 1]))
    C = X*X.T/3
    equal("PTQ: calibration second moment", C, M([[R(13, 27), R(5, 27)], [R(5, 27), R(61, 108)]]))
    loss_reconstruct = squared_frobenius(Ew*X)/6
    equal("PTQ: reconstruction value", loss_reconstruct, R(73, 31104))
    equal("PTQ: trace identity", loss_reconstruct, sp.trace(Ew*C*Ew.T)/2)
    e1, e2 = sp.symbols("e1 e2", real=True)
    e = M([[e1, e2]])
    row_loss = (e*C*e.T)[0]/2
    equal("PTQ: row Hessian", sp.hessian(row_loss, (e1, e2)), C)
    C_sensitive = M([[1, 1], [10, -10]])*M([[1, 1], [10, -10]]).T/2
    equal("PTQ: anisotropic second moment", C_sensitive, sp.diag(1, 100))

    residual = yq-target
    loss = squared_frobenius(residual)/2
    grad_W = residual*xq.T  # Declared clipped STE; all entries are interior here.
    grad_x = Wq.T*residual
    equal("QAT: quantized forward loss", loss, R(13, 128))
    equal("QAT: full precision comparison loss", squared_frobenius(y-target)/2, R(145, 2592))
    equal("QAT: weight surrogate gradient", grad_W, M([[R(9, 32), R(9, 32)], [R(-3, 16), R(-3, 16)]]))
    equal("QAT: activation surrogate gradient", grad_x, M([R(5, 16), R(-3, 8)]))
    W_new = W-R(1, 4)*grad_W
    equal("QAT: updated master weights", W_new, M([[R(119, 128), R(-73, 128)], [R(73, 192), R(137, 192)]]))
    equal("QAT: changed quantized weights", quant.matrix(W_new), M([[1, R(-1, 2)], [R(1, 2), R(3, 4)]]))
    y_new = quant.matrix(W_new)*xq
    equal("QAT: next forward output", y_new, M([R(3, 8), R(15, 16)]))
    equal("QAT: next forward loss", squared_frobenius(y_new-target)/2, R(37, 512))
    true("QAT: loss reduction in this one fixture", squared_frobenius(y_new-target)/2 < loss)
    w, h = R(1, 3), R(1, 100)
    scalar_loss = lambda t: (quant(t)-1)**2/2
    equal("STE: scalar true local finite difference", (scalar_loss(w+h)-scalar_loss(w-h))/(2*h), 0)
    equal("STE: scalar surrogate gradient", quant(w)-1, R(-3, 4))
    equal("STE: small step leaves forward unchanged", scalar_loss(w-R(1, 100)*(quant(w)-1)), scalar_loss(w))
    equal("scale: surrogate derivative at fixture", quant.encode(w)-w/quant.delta, R(-1, 3))
    equal("scale: true local scale finite difference", (UniformQuantizer(quant.delta+R(1,1000))(w)-UniformQuantizer(quant.delta-R(1,1000))(w))/R(2,1000), 1)

    alpha, delta, base = sp.symbols("alpha delta base", real=True)
    point = base+alpha*delta
    mean = (1-alpha)*base+alpha*(base+delta)
    variance = (1-alpha)*(base-point)**2+alpha*(base+delta-point)**2
    equal("stochastic: conditional mean identity", mean, point)
    equal("stochastic: conditional variance identity", variance, alpha*(1-alpha)*delta**2)
    equal("storage: payload and scales in bytes", R(1024*4+8*16, 8), 528)
    equal("storage: ideal compression ratio", R(1024*16, 1024*4+8*16), R(128, 33))
    true("accumulator: six narrow-signed 4-bit products fit signed 10 bits", 6*7*7 <= 2**9-1)
    true("accumulator: same worst case need not fit signed 9 bits", 6*7*7 > 2**8-1)
    print(f"PASS: {len(CHECKS)} exact planning-fixture checks; SymPy {sp.__version__}.")
    return len(CHECKS)


if __name__ == "__main__":
    run()
