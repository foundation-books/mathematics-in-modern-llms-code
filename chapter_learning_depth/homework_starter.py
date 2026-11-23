"""Chapter 11 inputs and helpers; complete Tasks A-F below."""
import sys
import sympy as sp

Q = sp.Rational
x0 = sp.Matrix([Q(2, 3), Q(5, 6)])
A = sp.Matrix([[0, Q(1, 2)], [0, 0]])
B = sp.Matrix([[0, 0], [-Q(1, 3), 0]])
K0 = sp.Matrix([[1, 2], [3, 4]])


def require_positive_square(K):
    if K.rows == 0 or K.rows != K.cols:
        raise ValueError("Expected a nonempty square matrix.")
    if any(entry.is_positive is not True for entry in K):
        raise ValueError("Entries must be provably positive.")


def sinkhorn_sweep(K):
    """One column-then-row sweep; preserve input precision."""
    K = sp.Matrix(K)
    require_positive_square(K)
    cols = [sum(K[:, j]) for j in range(K.cols)]
    C = K * sp.diag(*[1 / s for s in cols])
    rows = [sum(C[i, :]) for i in range(C.rows)]
    return sp.diag(*[1 / s for s in rows]) * C


def marginal_residual_vectors(M):
    M = sp.Matrix(M)
    if M.rows == 0 or M.rows != M.cols:
        raise ValueError("Expected a nonempty square matrix.")
    ones = sp.ones(M.rows, 1)
    return M * ones - ones, ones.T * M - ones.T


def is_symbolic_zero(expr):
    """Unresolved simplification is not a proof of inequality."""
    expr = sp.sympify(expr)
    return sp.simplify(sp.cancel(expr.rewrite(sp.exp))) == 0


if __name__ == "__main__":
    print("Python:", sys.version.split()[0])
    print("SymPy:", sp.__version__)
    # Complete Tasks A-F using these exact inputs and helpers.
