# Chapter 3: exact symbolic laboratory

For shared installation and chapter selection, see the
[companion guide](../README.md).

Python 3 and SymPy are the only dependencies. The checked environment uses
SymPy 1.13.3; this is a correctness demonstration, not a model experiment.
No network, GPU, trained weights or dataset is used during execution.

Run from the companion repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_tokens_embeddings/homework_starter.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_tokens_embeddings/symbolic_checks.py
```

The chapter prints the full assignment. `homework_starter.py` reproduces its
short scaffold; students supply their own matrices and solutions. The separate
`symbolic_checks.py` is an **instructor reference solution**: it independently
differentiates all 32 embedding variables, proves arbitrary-angle rotation
identities, checks all exact fixtures, and diagnoses both deliberate faults.
It also verifies the merge counts, logical costs, lookup adjoint, and scalar
causal-mask counterexample needed by the exercises. The final line announces
that all checks passed. Expected values are frozen in `reference_inputs.py`.

The residual checker reports “not certified zero” if simplification fails.
That is a verification failure, not a mathematical disproof; assumptions or
an independently justified transformation may be needed. A score disagreeing
with a separately derived exact fixture supplies an actual counterexample to
an implementation's stated convention. Norms and common shifts alone are
insufficient to distinguish opposite rotation directions.

The book's Exercise Answers appendix gives worked solutions to the written exercises.

The learned-spectrum extension in Part E checks the log-frequency derivative,
initialization, pure-rotation norms and common shifts, and the independent
reference/scaled/learned fixture. Its exact learned derivative is `-pi/2`.
