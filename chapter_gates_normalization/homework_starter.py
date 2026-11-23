import sys
import sympy as sp

Q = sp.Rational
x0 = sp.Matrix([Q(2, 3), Q(5, 6), 1, -1])
eps0 = Q(1, 144)
Wg = sp.Matrix([[1, 0], [0, 1], [0, 0], [0, 0]])
Wu = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
Wd = sp.Matrix([[1, 0, 1, 0], [0, 1, 0, 1]])

def core_norm(x: sp.MatrixBase, eps=sp.S.Zero,
              center: bool = False) -> sp.Matrix:
    """Real-vector map; callers must respect its domain."""
    x = sp.Matrix(x)
    eps = sp.sympify(eps)
    if x.cols != 1 or x.rows == 0:
        raise ValueError("Expected a nonempty column vector.")
    if eps.is_nonnegative is False:
        raise ValueError("epsilon must be nonnegative.")
    n = x.rows
    norm_input = x - sp.ones(n, 1) * sum(x) / n if center else x
    denom_sq = sp.simplify(norm_input.dot(norm_input) / n + eps)
    if denom_sq.is_zero is True:
        raise ValueError("Undefined zero-stabilizer input.")
    return norm_input / sp.sqrt(denom_sq)

def sigmoid(t):
    return 1 / (1 + sp.exp(-t))

def silu(t):
    return t * sigmoid(t)

def gated(h_norm: sp.MatrixBase) -> sp.Matrix:
    h_norm = sp.Matrix(h_norm)
    if h_norm.shape != (4, 1):
        raise ValueError("Expected a (4, 1) fixture vector.")
    u_g, u_v = Wg.T * h_norm, Wu.T * h_norm
    return Wd.T * u_g.applyfunc(silu).multiply_elementwise(u_v)

def zero_matrix_test(M: sp.MatrixBase) -> bool:
    # Unresolved simplification is not a disproof.
    return all(sp.simplify(entry) == 0 for entry in M)

xs = sp.Matrix(sp.symbols("x1:5", real=True))
eps = sp.symbols("epsilon", positive=True)
print("Python version:", sys.version.split()[0])
print("SymPy version:", sp.__version__)
# Recommended: A, B and D1–D4. C, D5, E and F are separate extensions.
# D5 uses the head, logits and projection from Exercise 10.4.
