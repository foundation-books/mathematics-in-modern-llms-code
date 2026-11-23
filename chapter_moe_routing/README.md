# Chapter 12 — exact sparse expert laboratory

Choose routing (A, B, C1, F1) or optimization and geometry (C2, D, E).
F2 is a separate equivalence check. The reference enumerates all 27
two-expert assignments in D1 and checks the six balanced survivors;
independent Top-2 is a separate comparison.

Use [expansion_student.py](expansion_student.py) for the unsolved A–F
interfaces. The printed homework gives the inputs and requirements.
[reference.py](reference.py) checks the foundation calculations and
[expansion_reference.py](expansion_reference.py) checks further cases,
including deliberately incorrect implementations.

For shared installation and chapter selection, see the
[companion guide](../README.md).

This layer-local rational fixture verifies the manuscript's teaching examples.
It is not a preceding decoder output, production router, histogram controller,
training run, or performance benchmark.

Run from the companion repository root with Python 3.10 or later and SymPy.

```sh
python3 chapter_moe_routing/expansion_student.py
python3 chapter_moe_routing/reference.py
python3 chapter_moe_routing/expansion_reference.py
python3 check.py --chapter 12
```

`fixture.py` supplies all exact data and the concrete-score Top-k helper.
`student_setup.py` exposes those inputs without solving the assignment.
`reference.py` is an instructor/reference implementation of the mathematical
checks. `test_reference.py` contains eight behavioral groups covering exact
outputs, wrong-weight detection, record identity, frozen-row independence,
exhaustive assignment, quantile ties/endpoints, the controller timeline and
normalization placement.

Code IDs are zero-based: code expert 0 is printed expert 1. Selected tuples
retain execution order; displayed sets are sorted separately. Linear expert
outputs include shared x/2, scale one and no residual. Counts describe selected
routes before any admission or dropping policy.

The exact `quantile_candidate` helper handles an interior integer quota using
the lower endpoint order statistic. It follows the reported operation order
but does not implement histogram estimation, reproduce a production controller,
or guarantee balanced row-wise reconstruction. In the fixture its first
candidate from zero retains loads (3,2,1). The k=E branch has no cutoff.

In the constructor domains, `N_route` counts total routes and `N` counts
rows in the routing group. The chapter gives the general proofs. The
reference sweeps finite constructor histograms and allocations.

## DeepSeek-informed mathematical extensions, 15 October 2026

These earlier extensions cover group-limited expert selection, replica
thresholds, packing/locality and negative cases. Use the current printed
routes for submission requirements.
All previous foundation and seven-addition regressions remain active. The
student starter contains inputs and unsolved interfaces; it imports no
instructor oracle. `extensions_reference.py` is a separate completed reference.

From the companion repository root run:

```sh
python3 chapter_moe_routing/extensions_reference.py
python3 -m unittest discover -s chapter_moe_routing -p test_extensions.py -v
```

`python3 check.py --chapter 12` also runs the old/new references and tests.
The independent oracles and deliberately faulty variants check finite exact
constructions; they do not establish model quality, production parity or GPU
performance.
