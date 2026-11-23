# Chapter 1 symbolic laboratory

For shared installation and chapter selection, see the
[companion guide](../README.md).

Requires Python 3.10+ and SymPy. Tested with Python 3.12.3 and SymPy 1.13.3.
No GPU, external data, downloaded model, or remote service is used.
Install the tested version with `python3 -m pip install sympy==1.13.3`.

Run from the companion repository root:

```sh
python3 chapter_language_modeling/symbolic_warmup.py
python3 chapter_language_modeling/ch01_homework_starter.py
python3 chapter_language_modeling/ch01_examples.py
python3 -m unittest discover -s chapter_language_modeling -p 'test_*.py'
```

`symbolic_warmup.py` first prints `Matrix([[-1/4], [1/4]])`: differentiate
the two-logit loss before substituting `(log(3), 0)`, then compare with
softmax minus the target. Use A–C as a model/scoring milestone, D as
information measures and E as generation/coding; all remain required.

`ch01_homework_starter.py` is the student scaffold; copy it to your submitted
script or notebook and complete Parts A–E. `ch01_examples.py` is the complete
reference solution and acceptance checks. Read it after attempting the tasks.
The student chapter contains the assignment, exact input tables and conventions.
The book's Exercise Answers appendix gives worked solutions to the written exercises.

The reference derives every quantity from exact inputs and verifies real-logit
identities and a four-category positive-weight decomposition. Unknown support
is rejected until instantiated or made positive by explicit assumptions.
Exact boundary distributions use separate zero/infinity branches. In masked
scoring only selected losses are evaluated; observed prefixes are preserved.

The checks execute 21 deliberately invalid inputs and 40 encode/decode round
trips. Six behavioral tests also cover excluded impossible targets, malformed
bitstrings, default contexts, fresh scoring, and unknown symbolic support.
These computations verify the stated examples and identities; the manuscript
contains the general proofs. They do not infer an empirical target law or
measure real-model quality.

Submit a fresh-run notebook or script plus a short report containing exact
outputs, assumptions, derivations, interpretations and a test-results summary.
