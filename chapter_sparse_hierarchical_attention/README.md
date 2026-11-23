# Chapter 7 exact attention laboratory

For shared installation and chapter selection, see the
[companion guide](../README.md).

Start with A–D (dense, restricted, causal and hierarchical comparisons).
E1/F1/F3 extend the generic lab; E2/F2 use the separate printed QSA
inputs. The written exercises use query 8 unless specified otherwise.

This companion uses Python, SymPy, and the standard library. It demonstrates
specified mathematical operators, not model quality or execution speed.
`student_setup.py` supplies inputs only; `reference.py` supplies the instructor
operators. `homework_reference.py` runs the complete A–F computation and
`test_reference.py` contains ten behavioral groups.
Query/key/value vectors are columns, K/V store rows, and external
positions and blocks use one-based IDs. The router computes mean keys and
scaled dot products; it does not substitute a fixture-only geometric shortcut.

From the companion repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_sparse_hierarchical_attention/student_setup.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_sparse_hierarchical_attention/homework_reference.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s chapter_sparse_hierarchical_attention -p 'test_*.py' -v
```

The reference target in the supplied plan/documentation is SymPy 1.14.0.
Execution was verified with Python 3.12.3 and SymPy 1.13.3, which is already
installed; record your actual versions. `requirements.txt` declares the
supported minimum rather than claiming the installed runtime is 1.14.0.
No network, external data, training, model download or GPU is needed.

The checks cover exact outputs, all eight queries, each earlier suffix
invariance test, deliberate future leakage, request reset/padding exclusion,
probability/error identities, exhaustive small oracles, pooling, hierarchy,
reuse, auxiliary gradients and pair-count sweeps. General proofs remain in
the manuscript. A finite suite cannot prove all-input causality, latency or
downstream task accuracy. Full sorting is the reference top-k algorithm;
undecidable symbolic score order is rejected.

Finite-entry validation scans the supplied matrices, and SymPy simplification
adds work beyond the chapter's score-pair counts. These helpers are correctness
references. The pooling helper treats supplied K/V rows as the legal prefix;
slice away future positions and padding before an earlier query.
