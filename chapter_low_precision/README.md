# Chapter 16 exact quantization laboratory

The reference also prints the optional two-column reconstruction comparison
from Section 15.5 and Exercise 15.6. Its stipulated rank-one candidates use
exact rationals and are separate from the default quantizer and Part C's
unchanged three-column calibration data. Tests check the symbolic comparison,
both strict orderings, the tie and the numerical losses 25 and 1/400.

For shared installation and chapter selection, see the
[companion guide](../README.md).

CPU-only Python 3.10+ and SymPy. Verified here with Python 3.12.3, SymPy
1.13.3, and pytest 8.3.4; the plan's original reference run used 1.14.0.
No installation, model download, training corpus, or hardware experiment is
required. All arithmetic is exact reconstructed integer/rational arithmetic.

Run from the companion repository root:

```sh
python3 chapter_low_precision/homework_student.py
python3 chapter_low_precision/fixture_checks.py
python3 chapter_low_precision/reference.py
python3 -m pytest -q chapter_low_precision/test_reference.py
```

The starter exposes input shapes and versions. Implement the interfaces
for your assigned route: integer (A1, B, E1–E2, F1) or floating codebook
(A2, E3, F2), with C–D shared. Use the printed quantizer directly for
the integer reconstructions in C. Follow the derivations and
interpretations stated in the chapter.

`fixture_checks.py` is the plan's unchanged 73-check oracle. `reference.py`
adds the full six-part calculation, legal decoding, zero-tensor scale policy,
strict STE masks, scale-surrogate branches, row Hessian, metadata/width ledger,
and exact acceptance checks. The test oracle independently uses standard
library Fraction arithmetic and nearest-even rounding across 435 encoding
cases (145 inputs under three configurations), including clipping and affine
zero points. These finite witnesses do not prove the general error theorems.

The reference uses explicitly defined surrogate gradients. The changed
attention logits represent an analytic perturbation.
Storage metadata precision is an accounting assumption; its exact rational
scales are not a simulation of rounding a scale to a hardware dtype.

## DeepSeek-informed mathematical extensions, 15 October 2026

The current printed homework extends Parts A, E and F with all E2M1 encodings/ties, complete floating master update and group-scaled arithmetic.
All previous foundation and seven-addition regressions remain active. The
student starter contains inputs and unsolved interfaces; it imports no
instructor oracle. `extensions_reference.py` is a separate completed reference.

From the companion repository root run:

```sh
python3 chapter_low_precision/extensions_reference.py
python3 -m unittest discover -s chapter_low_precision -p test_extensions.py -v
```

`python3 check.py --chapter 16` also runs the old/new references and tests.
The independent oracles and deliberately faulty variants check finite exact
constructions; they do not establish model quality, production parity or GPU
performance.
