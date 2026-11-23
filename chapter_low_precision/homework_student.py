import sys
import sympy as sp
print("Python:", sys.version.split()[0])
print("SymPy:", sp.__version__)
R, M = sp.Rational, sp.Matrix
x = M([R(2, 3), R(5, 6)])
W = M([[1, R(-1, 2)], [R(1, 3), R(2, 3)]])
V = M([[1, 0], [0, 1], [1, 1]])
ell = M([0, sp.log(2), sp.log(3)])
target = M([0, 1])
delta, qmin, qmax, zero = R(1, 4), -7, 7, 0
def round_even(value):
    raise NotImplementedError("Complete A1")
def encode(value, delta, qmin=-7, qmax=7, zero=0):
    raise NotImplementedError("Complete A1")
def decode(code, delta, zero=0):
    raise NotImplementedError("Complete A1")
assert W.shape == (2, 2) and x.shape == (2, 1)
# Integer: A1, B, E1–E2, F1. Floating: A2, E3, F2. C–D are shared.


# Separate E2M1 readout from the printed floating-codebook section.
fp4_masters = (R(1, 5), R(7, 10), R(11, 10), R(14, 5))
fp4_input, fp4_target = (1, 1, 1, 1), 4
fp4_scale, fp4_rate = R(1, 2), R(1, 10)

def decode_e2m1(code):
    raise NotImplementedError("A2: decode all sixteen bit patterns")

def encode_e2m1(value, scale):
    raise NotImplementedError("A2: finite inputs, positive scale and fraction-bit ties")

def fp4_master_step(masters, inputs, target, scale, rate):
    raise NotImplementedError("E3: declare STE, update masters, regenerate payloads")
