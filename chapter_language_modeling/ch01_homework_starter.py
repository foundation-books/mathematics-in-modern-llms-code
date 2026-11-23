"""Student scaffold for the single Chapter 1 SymPy homework.

Implement Parts A–E yourself. This starter demonstrates one symbolic check;
the complete reference laboratory is a separate solution file.
"""

import platform
import sympy as sp
print("Python:", platform.python_version())
print("SymPy:", sp.__version__)
R = sp.Rational
vocabulary = ("A", "B", "C", "<eos>")
observed = ("A", "B", "<eos>")
all_targets = (1, 1, 1)
completion_targets = (0, 1, 1)
rows = {
    (): (R(1,2), R(1,4), R(1,8), R(1,8)),
    ("A",): (R(1,8), R(1,4), R(1,2), R(1,8)),
    ("A", "B"): (R(1,8), R(1,8), R(1,4), R(1,2)),
}
default_row = (R(1,4),) * 4
z = sp.symbols("z0:4", real=True)
c = sp.symbols("c", real=True)
normalizer = sum(sp.exp(value) for value in z)
probabilities = tuple(sp.exp(v)/normalizer for v in z)
assert sp.simplify(sum(probabilities) - 1) == 0

# Part A: validated lookup, complete sequences, masks and invalid inputs.
# Part B: shift/target-loss checks, justified by finite real logits; toy subs.
# Part C: selected scoring. Keep every observed prefix, even excluded targets.
# Part D: entropy/CE/KL. Skip zero target mass before logs; branch on support.
# Part E: temperatures, the two-step tree, and changing-codebook decoding.
