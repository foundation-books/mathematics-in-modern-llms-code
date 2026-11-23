import sympy as sp

def rotation(phi):
    c, s = sp.cos(phi), sp.sin(phi)
    return sp.Matrix([[c, -s], [s, c]])

def certify_zero_matrix(residual, label):
    r = residual.applyfunc(lambda e: sp.simplify(sp.trigsimp(e)))
    if r != sp.zeros(*r.shape):
        raise AssertionError(f"{label}: not certified zero:\n{r}")

alpha, beta = sp.symbols("alpha beta", real=True)
omega0, omega1 = sp.pi / 2, sp.pi / 4
scale = sp.Integer(2)
print("SymPy version:", sp.__version__)
