"""Student starter for Chapter 2: Matrix Derivatives and Gradient Flow.

A: construct the head. B: compare scalar differentiation and matrix rules.
C: evaluate both exact fixtures. D: differentiate with an independent target,
then substitute its value. Add forward expressions and derivatives yourself.
The historical ch03 filenames preserve authoring identity, not print numbering.
"""

import platform

import sympy as sp

N, d, N_vocab = 2, 2, 3
xs = sp.symbols("x00 x01 x10 x11", real=True)
ws = sp.symbols("w00 w01 w02 w10 w11 w12", real=True)
bs = sp.symbols("b0 b1 b2", real=True)
X, W, b = sp.Matrix(2, 2, xs), sp.Matrix(2, 3, ws), sp.Matrix(bs)
T = sp.Matrix([[1, 0, 0], [0, 0, 1]])
X_fixture = sp.Matrix([[1, 1], [0, 1]])
W_fixture = sp.Matrix([[sp.log(2), -sp.log(3), 0], [0, sp.log(3), 0]])
b_fixture = sp.zeros(3, 1)
zero_weights = sp.zeros(2, 3)

# Part D: keep q independent until after differentiation.
A_entries = sp.symbols("a00 a01 a10 a11", real=True)
A = sp.Matrix(2, 2, A_entries)
a, b_shared = sp.Matrix([1, 0]), sp.Matrix([1, 1])
q = sp.Matrix(sp.symbols("q0 q1", real=True))
A_fixture = sp.Matrix([[1, 2], [0, 1]])

assert X.shape == (2, 2) and W.shape == (2, 3) and b.shape == (3, 1)
if __name__ == "__main__":
    print(f"Setup ready: Python {platform.python_version()}, SymPy {sp.__version__}.")
    print("Parts A--D: head, independent derivatives, exact fixtures, detached target.")
