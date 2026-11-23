"""Student starter: complete the six printed homework parts before opening the reference."""
import sys
import sympy as sp
R = sp.Rational
Z = sp.Matrix([R(2,3),R(5,6)])
PAIRS = [(1,2)]*4 + [(1,3)]*2 + [(1,4),(2,3),(2,4),(3,4)]
COEFFICIENTS = (R(1,3),R(2,3))
HOMES = {1:0,2:0,3:1,4:1}

def part_a():
    """Existing routing foundation. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part A")

def part_b():
    """Immutable route construction. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part B")

def part_c():
    """Constructive balancing. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part C")

def part_d():
    """Lower bounds and exact oracles. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part D")

def part_e():
    """Forward and backward ownership. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part E")

def part_f():
    """Faults and interpretation. See the printed part for complete required outputs."""
    raise NotImplementedError("Complete Part F")


# Additional printed group-selection and persistent-copy inputs (no solutions).
GROUP_SCORES = {1: 10, 2: 1, 3: 9, 4: 8}
EXPERT_GROUPS = {1: (1, 2), 2: (3, 4)}
GROUP_K, GROUP_BUDGET = 2, 1
GROUP_MAPS = (sp.diag(1, 0), sp.diag(0, 1), sp.eye(2), sp.diag(2, 1))
EXPERT_WORK = (70, 60, 50, 40)
REPLICA_SLOTS, DEVICE_CAPACITIES = 8, (2, 2, 2, 2)
PACKING_COUNTEREXAMPLE = (8, 7, 6, 5, 4, 2)

def group_router(scores, groups, k, group_budget):
    raise NotImplementedError("Extend Part A with exact selection and subset oracle")

def replica_plan(work, slots, capacities, node_constraints=None):
    raise NotImplementedError("Extend Part D with thresholds, packing and feasibility")

if __name__ == "__main__":
    print("Python",sys.version.split()[0],"SymPy",sp.__version__)
    print("Inputs loaded; implement Parts A-F. No solutions run here.")
