# Chapter 14 optimizer laboratory

The optimizer lab (A–B) and QK-controller lab (C–F) can be assigned
separately. For F's no-clipping comparison, use the printed optimizer
recipe; completed A–B tables are unnecessary.

Start with [expansion_student.py](expansion_student.py) and follow the
printed A–F homework. [reference.py](reference.py) checks the foundation
calculations and [expansion_reference.py](expansion_reference.py) checks
the additional cases. In the shared content/rotary fixture, both branches
are scalar. The symmetric stored product factor `gamma=1/4` gives
per-projection factors of `1/2`.

For shared installation and chapter selection, see the
[companion guide](../README.md).

Run all commands from the companion repository root. Python and SymPy are the only
dependencies; the reference was tested with Python 3.12.3 and SymPy 1.13.3.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_optimization/expansion_student.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_optimization/reference.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_optimization/expansion_reference.py
python3 check.py --chapter 14
```

The starter supplies the same exact fixtures as the printed assignment.
`reference.py` is an executable mathematical reference. Its meaningful
checks cover symbolic loss differentiation, moment corrections, outside-root
epsilon, decoupled decay, supported factors, rectangular transposes, zero and
rank loss, cubic identities, practical-quintic nonconvergence to one, scale,
head coupling, missing gradients, tied ownership and step counters. The
behavioral tests add nonzero-parameter chain rule, orthogonal-basis equivariance,
norm-ball attainment, ownership/error cases, a negative polar-magnitude
certificate, and the complete Task F comparison. There are 93 reference checks
and ten behavioral tests.

The Task F table includes SGD, first Adam, exact supported polar, five-step
cubic and five-step quintic. Each row separates the raw direction, declared
scale, nominal direction, loss update, independent old-parameter decay and
resulting first parameter. The output declares the momentum, normalization,
epsilon, grouping, precision and decay conventions. Every supplied polar
fixture also checks that its symmetric reconstruction magnitude is positive
semidefinite; reconstruction and support Grams alone do not certify that sign.

The finite comparison table exhausts the specified domain: six fixtures
(batch, wide, tall, deficient, token and zero), two polynomials and counts
0, 1, 2, 5, 10, for **60 rows**. Calculations use 80-digit SymPy evaluation;
the arithmetic-agreement tolerance is 1e-30. A finite iteration's error need
not meet that tolerance. A printed zero can reflect rounding at the chosen
precision rather than equality of a finite iterate with its exact limit.

The two polynomials share Frobenius normalization with no epsilon to isolate
their scalar dynamics. Quintic coefficients are exact rationals in the
symbolic check. This does not emulate the pinned source's BF16 arithmetic
or its 1e-7 safeguard. The baseline has ordinary mu=1/2 EMA, no clipping,
five cubic steps, sqrt(max(m,n)) scaling, eta=1/10 and lambda=1/5 independent
old-parameter decay. The exact variant uses the supported polar instead.

`Parameter` implements the optional absent-gradient policy: `None` skips
state/count/decay/update; a zero matrix advances state and decay. Adam counts
actual state updates. Tied uses share one Parameter object, sum gradients
and update once. Distinct owners of one value-storage object are rejected.
The small real polar routine uses certified rank-one/2-by-2 formulas and a
condensed SVD for the other exact fixtures, rather than a production solver.

All outputs are algebraic diagnostics, with no model training, performance
measurement, GPU or external data. Do not interpret them as optimizer-quality
or wall-clock results.
