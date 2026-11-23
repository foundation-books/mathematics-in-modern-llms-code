"""Student starter: complete the six printed homework parts before opening the reference."""
import sys
import sympy as sp
R = sp.Rational
Z = sp.Matrix([R(2,3),R(5,6)])
W1 = sp.Matrix([[1,2],[3,4],[-1,1],[2,0]])
W2 = sp.Matrix([[1,0,2,-1],[0,1,1,2]])
PROBE_SEED = sp.Matrix([1,2])

def part_a():
    """Original attention oracle. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part A")

def part_b():
    """Forward layouts. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part B")

def part_c():
    """Backward and adjoints. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part C")

def part_d():
    """Global objective. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part D")

def part_e():
    """Optimizer and expert ownership. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part E")

def part_f():
    """Integrated failures and numerical scope. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part F")


# Two complete printed schedules; these are assignment inputs, not an executor.
PIPELINE_X, PIPELINE_TARGET = (4, 5), (1, 2)
PIPELINE_A, PIPELINE_B, PIPELINE_RATE = R(1, 2), 2, R(1, 100)
SCHEDULE_I = (
    (("F", 0, 0), None), (("F", 0, 1), ("F", 1, 0)), (None, ("D", 1, 0)),
    (("D", 0, 0), ("W", 1, 0)), (("W", 0, 0), ("F", 1, 1)),
    (None, ("D", 1, 1)), (("D", 0, 1), ("W", 1, 1)), (("W", 0, 1), None))
SCHEDULE_D = (
    (("F", 0, 0), None), (("F", 0, 1), ("F", 1, 0)), (None, ("D", 1, 0)),
    (("D", 0, 0), ("F", 1, 1)), (("W", 0, 0), ("D", 1, 1)),
    (("D", 0, 1), ("W", 1, 0)), (("W", 0, 1), ("W", 1, 1)))

def check_pipeline(rows, saved_records, optimizer_boundary):
    raise NotImplementedError("Extend Part D: typed readiness, resources, versions and lifetimes")

def execute_pipeline(rows):
    raise NotImplementedError("Extend Part E: complete gradients and one owner update")

if __name__ == "__main__":
    print("Python",sys.version.split()[0],"SymPy",sp.__version__)
    print("Inputs loaded; implement Parts A-F. No solutions run here.")
