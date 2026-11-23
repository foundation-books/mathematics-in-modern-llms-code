# Chapter 11 SymPy laboratory

For shared installation and chapter selection, see the
[companion guide](../README.md).

This is an exact teaching laboratory for learning across depth. It uses
Python 3.10+ and SymPy, with Python 3.12.3 / SymPy 1.13.3 used for the executed
reference. No model weights, training, accelerator, data download, or network
service is required.

Run from the companion repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_learning_depth/homework_starter.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_learning_depth/chapter11_fixture.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_learning_depth/reference.py
```

`homework_starter.py` is printed verbatim in the final homework section and
contains all essential constants and helpers. Complete Tasks A–F below its
final comment. The starter prints versions but does not contain complete
exercise solutions. `chapter11_fixture.py` supplies initial rational checks.

`reference.py` is a separate instructor program. It independently differentiates
the residual and gated graphs, checks full/block source timing including
absent/cancelling zero sources and an incomplete last block, computes stream
updates, verifies selected powers, and analyzes finite positive-matrix scaling
and the dynamic Jacobian counterexample. It rejects malformed scaling inputs
and unresolved positivity. A 1×1 provably positive symbolic input is accepted.

Use `--json /tmp/chapter11-reference.json` to save diagnostic output outside
the repository. The numerical Sinkhorn table declares column-then-row order,
80 decimal digits and counts 1,2,5,10,20. Exact symbolic identities use
separate checks from numerical tolerances; 20 sweeps are sufficient only for
the demonstrated tolerance on this fixture. No universal convergence rate,
performance comparison, or whole-network stability claim is made.

The chapter gives the general proofs. The executable reference checks selected
exact and numerical cases.
