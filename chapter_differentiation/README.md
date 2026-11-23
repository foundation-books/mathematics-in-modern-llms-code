# Chapter 2: Matrix Derivatives and Gradient Flow

The [companion guide](../README.md) describes shared setup and chapter selection.
The historical `ch03` filenames identify the original authoring chapter. The
current chapter has eight exercises and one integrated four-part homework:

- A constructs the mean prediction head with independent real entries.
- B compares scalar symbolic differentiation with independently assembled matrix rules.
- C evaluates the zero-weight and exact log-valued fixtures.
- D differentiates with an independent target, then substitutes its value to model a stopped path.

The starter supplies data without computing answers. The instructor solution
and checks implement all four parts, including exact rational directional
checks with both branches recomputed and with the target frozen once.
Differentiate before substitution: the fixture's logarithms are values, not
relations among independent parameters. Assertions establish fixed-size
identities and fixture properties; the book supplies the general derivations.

Run from the companion repository root with Python 3.10 or later and SymPy 1.13.3:

```sh
python3 chapter_differentiation/ch03_sympy_homework.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_differentiation/ch03_instructor_checks.py --output-dir /tmp/mml-matrix-core
python3 check.py --chapter 2
```

Required checks use SymPy and the standard library. No numerical precision
sweep is part of the core homework. Actual versions are recorded in the
output; the validated environment is Python 3.12.3 and SymPy 1.13.3.
## Optional historical extensions

The full `6 x 13` logit Jacobian comparison verifies separately assembled
JVP and VJP formulas and their adjoint pairing. An optional precision
experiment preserves the former rational fixture, three directions, twelve
steps, ordinary floats and 80-digit mpmath arithmetic. It detects omitted-mean
and zero-bias faults independently. The former fixture is used only in this
extension; it is distinct from the core homework's exact log-valued fixture.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_differentiation/ch03_instructor_checks.py --extensions --output-dir /tmp/mml-matrix-extensions
```

This command additionally needs mpmath 1.3.0, pinned in `requirements.txt`.
It produces 72 numerical cases and four bug detections. The coordinates are
row-major entries of X, then W, then the three bias entries:
`x00 x01 x10 x11 w00 w01 w02 w10 w11 w12 b0 b1 b2`.
Directions are `(1,...,13)/sqrt(819)`, `(1,-1,...,1)/sqrt(13)` and `e12`;
`e12` means one-based coordinate 12, middle bias `b1`, Python index 11.
Every direction uses `h=10^-k`, `k=1,...,12`. Scalar symbolic derivatives
supply the references; a separate shifted scalar loss evaluator supplies the
finite differences. High-precision inputs come from exact rationals.

The CSV records backend, direction, step, pairing, finite difference and
absolute discrepancy. Interpret actual errors without prescribing an optimum
step or error curve. The simplified optional truncation argument in the
historical instructor notes assumes a third derivative bound and exact
subtraction/division; it is not a required written exercise. Symbolic and
numerical checks do not establish arbitrary-dimensional correctness or an
unspecified custom backward rule. Keep generated outputs outside Git.
