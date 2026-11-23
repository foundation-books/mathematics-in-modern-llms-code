"""Self-contained inputs for Chapter 14 Tasks A–F; complete the tasks below."""
import sys
import sympy as sp

q = sp.Rational
input_feature = sp.Matrix([q(2, 3), q(5, 6)])
input_feature2 = sp.Matrix([0, 1])
R = sp.Matrix([[1, 0], [0, 1], [0, 0], [0, 0]])
W0 = sp.zeros(2)
M = sp.Matrix([[3, 1], [1, 3]])
A = sp.Matrix([[3, 0, 0], [0, 1, 0]])
A0 = sp.Matrix([[3, 0, 0], [0, 0, 0]])
MH = sp.Matrix([[3, 0], [1, 1]])
wvars = sp.symbols("w11 w12 w21 w22", real=True)
W = sp.Matrix(2, 2, wvars)
print(sys.version.split()[0], sp.__version__)
# Complete Tasks A-F using these inputs.
