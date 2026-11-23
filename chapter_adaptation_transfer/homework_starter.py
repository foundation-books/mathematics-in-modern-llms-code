"""Inputs for merged Chapter 17's four-part homework, A--D.

Use exact arithmetic and explain each result. Symbols u and t additionally
satisfy u<1 and t<1; positivity alone does not encode those bounds.
The current manuscript contains every input and required written argument.
"""
import sympy as sp

R = sp.Rational
O_STAR = sp.Matrix([R(2, 3), R(5, 6)])
W_BASE = sp.Matrix([[1, 0], [1, 0], [1, 0]])
TEACHER = sp.Matrix([R(1, 2), R(1, 3), R(1, 6)])
A0, B0 = sp.Matrix([[1, 0]]), sp.zeros(3, 1)
ROOT_TEACHER = sp.Matrix([R(3, 4), R(1, 4)])
UNIFORM_BINARY = sp.Matrix([R(1, 2), R(1, 2)])
PI_REF = TEACHER.copy()  # A separate preference learning case.
REWARDS = sp.Matrix([sp.log(2), 0, -sp.log(2)])
CURRENT_FACTORS = ((R(1, 2), R(1, 2)), (R(1, 4), R(1, 2), R(1, 2)))
REFERENCE_FACTORS = ((R(1, 2), R(1, 4)), (R(1, 4), R(2, 3), R(3, 4)))
u, t = sp.symbols("u t", positive=True)  # Additional domain: u,t < 1.


def softmax(logits):
    """Return a normalized column for finite logits."""
    raise NotImplementedError("Implement softmax.")


def kl_interior(averaging_law, predictor):
    """Use aligned, positive probability columns."""
    raise NotImplementedError("Implement the directed finite KL sum.")


def assert_zero(residual):
    """Fail visibly unless every scalar or shaped-matrix residual is zero."""
    raise NotImplementedError("Implement exact residual checking.")


if __name__ == "__main__":
    print(f"SymPy {sp.__version__}. Complete the helpers and Parts A--D.")
