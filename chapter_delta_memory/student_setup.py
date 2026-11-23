"""Chapter 9 inputs only. Implement the assignment before consulting reference.py."""
import sys
import sympy as sp

R = sp.Rational
I = sp.eye(2)
eA, eB = sp.Matrix([1, 0]), sp.Matrix([0, 1])
f1, f2 = sp.Matrix([1, 0]), sp.Matrix([0, 1])
S0 = sp.zeros(2, 2)
keys = [eA, eB, eA]
values = [f1, f2, f2]
queries = [eA, eB, eA]
betas = [sp.Integer(1), sp.Integer(1), R(1, 2)]
gates = [I, I, sp.diag(R(1, 2), R(3, 4))]


def gated_chunk_fixture():
    initial = sp.Matrix([[2, 4], [6, 8]])
    overlap = sp.Matrix([R(3, 5), R(4, 5)])
    ks = [eA, overlap, eB]
    vs = [sp.Matrix(x) for x in ([3, 1], [R(16, 5), R(19, 5)], [-R(49, 40), R(43, 20)])]
    qs = [eA, eB, overlap]
    bs = [R(1, 2)] * 3
    ds = [sp.diag(R(1, 2), R(3, 4)), sp.diag(R(1, 2), R(1, 4)), sp.diag(R(1, 3), R(1, 2))]
    return initial, ks, vs, qs, bs, ds


def exact_equal(left, right):
    """True when all explicit matrix residuals simplify to zero."""
    return left.shape == right.shape and all(
        sp.simplify(entry) == 0 for entry in left - right
    )


if __name__ == "__main__":
    print("Python:", sys.version.split()[0], "SymPy:", sp.__version__)
    print("State:", S0.shape, "events:", len(keys))
