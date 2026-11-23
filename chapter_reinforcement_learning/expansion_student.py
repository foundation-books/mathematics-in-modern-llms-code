"""Chapter 18 starter. Complete A1–A8, then choose B, C–E, or F–G."""
import sys
import sympy as sp
R = sp.Rational
Z = sp.Matrix([R(2,3),R(5,6)])
LOCAL_BOUNDS = (-R(1,5),R(1,5))
ADVANTAGE = R(1,4)
RESTORING = R(1,10)
ADMISSION = (1,R(1,2),R(1,4))

def part_a():
    """Work through the eight foundation tasks in the printed homework."""
    raise NotImplementedError("Complete Part A")

def part_b():
    """Enumerate the group law and compare the estimator means."""
    raise NotImplementedError("Complete Part B")

def part_c():
    """Compare the effort-dependent length charge and policy movement geometry."""
    raise NotImplementedError("Complete Part C")

def part_d():
    """Compare local derivative rules with their stated stop-gradient conventions."""
    raise NotImplementedError("Complete Part D")

def part_e():
    """Apply a parameter step and compute the new normalized probabilities."""
    raise NotImplementedError("Complete Part E")

def part_f():
    """Compute attempt and admission laws and corrected expectations."""
    raise NotImplementedError("Complete Part F")

def part_g():
    """Use the specified two-token/one-token cut in the repacking audit."""
    raise NotImplementedError("Complete Part G")

if __name__ == "__main__":
    print("Python",sys.version.split()[0],"SymPy",sp.__version__)
    print("Inputs loaded for the foundation and selectable investigations.")
