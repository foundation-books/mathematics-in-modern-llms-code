"""Input-only setup for the six Chapter 7 homework tasks."""
import sys
import sympy as sp

exponential_weights = tuple(map(sp.Integer, (9, 1, 4, 4, 1, 1, 2, 2)))
q = sp.Matrix([sp.sqrt(2), 0])
Q = sp.Matrix([list(q) for _ in exponential_weights])
K = sp.Matrix([[sp.log(w_exp), 0] for w_exp in exponential_weights])
V = sp.Matrix([(1, 0), (0, 0), (0, 1), (0, 1),
               (0, 0), (0, 0), (1, 1), (1, 1)])

if __name__ == "__main__":
    print("Python:", sys.version.split()[0], "SymPy:", sp.__version__)
    print("Q/K/V shapes:", Q.shape, K.shape, V.shape)
    print("Complete tasks A–F using the printed chapter; positions are one-based.")
