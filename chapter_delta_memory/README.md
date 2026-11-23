# Corrective memory laboratory

For shared installation and chapter selection, see the
[companion guide](../README.md).

One Chapter 9 assignment, Parts A–F. The chapter contains the complete student
statement; `student_setup.py` provides data and an equality helper without
solutions. `reference.py` and `test_reference.py` are instructor material.

Tested environment: Python 3.12.3, SymPy 1.13.3 (the pinned dependency).
No NumPy, GPU library, network, weights, dataset, or training is required.
Run from the companion repository root in an environment with SymPy:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_delta_memory/student_setup.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_delta_memory/reference.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s chapter_delta_memory -p 'test_*.py' -v
```

The reference derives a six-entry matrix gradient and two general-key identities
symbolically, computes the controlled fixtures, verifies characteristic and
composition identities, and compares every output and final state in independent
sequential, affine-prefix, and triangular organizations. The gated solve
covers the three-event fixture in all four partitions. The local pullback is
checked against symbolic differentiation of the scalar read and state loss.
The reference also covers all
16 contiguous partitions of a five-event rectangular case, including the
required `[2,3]`, `[3,2]`, and `[2,2,1]`, as well as all four partitions of both
three-event ungated cases. Zero key, zero correction, reset, zero retention
coordinate, non-unit key, overlap, nonzero incoming state, and short tails are
explicit cases. The checks detect swapped gate order, use of the final
state for earlier reads and a zero-state reset at a chunk boundary.

The diagonal gate is checked for shape and diagonality. Numeric range bounds
are mathematical assumptions, not restrictions on symbolic scalars accepted
by these teaching functions. `exact_equal` checks explicit residuals that
simplify to zero; unresolved symbolic expressions require reasoning. Exact
agreement is not a bitwise floating-point guarantee or production-model test.

Submit one runnable script/notebook, A–F explanations and test report with
versions, exact residuals, counterexample diagnoses and the 400–600-word
interpretation. The grading weights are 15/15/20/15/25/10, totaling 100.
