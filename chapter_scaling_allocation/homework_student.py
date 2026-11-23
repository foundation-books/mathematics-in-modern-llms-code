import sys
import sympy as sp

print("Python:", sys.version.split()[0])
print("SymPy:", sp.__version__)

n, d, c = sp.symbols("n d c", positive=True)
A, B, alpha, beta = sp.symbols(
    "A B alpha beta", positive=True)
tau = sp.symbols("tau", nonnegative=True)
loss = 1 + 4 / n + 9 / d

grid = [(sp.Integer(ni), sp.Integer(di))
        for ni in (4, 8, 16)
        for di in (9, 18, 36)]
X = sp.Matrix([[1, 1 / ni, 1 / di]
               for ni, di in grid])
y = sp.Matrix([loss.subs({n: ni, d: di})
               for ni, di in grid])

perturbed_y = y.copy()
changed_row = grid.index(
    (sp.Integer(4), sp.Integer(18)))
perturbed_y[changed_row] += sp.Rational(1, 50)
assert X.rows == 9 and X.cols == 3
assert len(y) == 9
# Add Tasks A-F with exact checks and explanations.
