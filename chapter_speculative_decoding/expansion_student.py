"""Student starter: complete the six printed homework parts before opening the reference."""
import sys
import sympy as sp
R = sp.Rational
Z = sp.Matrix([R(2,3),R(5,6)])
LANGUAGE = {b'{"ok":true}',b'{"ok":false}'}
TOKENS = {'START':b'{"ok":','T_FULL':b'true}', 'F_FULL':b'false}',
          'T':b'true','F':b'false','CLOSE':b'}','BAD':b'x','OVER':b'true}x'}
EOS = 'EOS'

def part_a():
    """Original sampler foundation. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part A")

def part_b():
    """Language interface. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part B")

def part_c():
    """Complete local masking. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part C")

def part_d():
    """Global conditioning. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part D")

def part_e():
    """Speculative state. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part E")

def part_f():
    """Contract violations. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part F")

if __name__ == "__main__":
    print("Python",sys.version.split()[0],"SymPy",sp.__version__)
    print("Inputs loaded; implement Parts A-F. No solutions run here.")
