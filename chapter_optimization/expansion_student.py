"""Student starter: complete the six printed homework parts before opening the reference."""
import sys
import sympy as sp
R = sp.Rational
Z = sp.Matrix([R(2,3),R(5,6)])
Q_ROWS = (sp.Matrix([[30,0]]),sp.Matrix([[12,0]]))
K_ROWS = (sp.Matrix([[30,0]]),sp.Matrix([[15,0]]))
THRESHOLD = sp.Integer(100)

def part_a():
    """Preserve the optimizer reference. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part A")

def part_b():
    """Partition and finite-transform geometry. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part B")

def part_c():
    """Derive QK contraction. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part C")

def part_d():
    """Shared-key paths. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part D")

def part_e():
    """Track observation endpoints. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part E")

def part_f():
    """Complete a controlled variant. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part F")

if __name__ == "__main__":
    print("Python",sys.version.split()[0],"SymPy",sp.__version__)
    print("Inputs loaded; implement Parts A-F. No solutions run here.")
