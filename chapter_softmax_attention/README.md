# Softmax attention companion

For shared installation and chapter selection, see the
[companion guide](../README.md).

One Chapter 4 homework, *Symbolic Verification of Softmax Attention*. The student
statement is fully contained in the book. `student_setup.py` supplies only the
data and independent symbols; `reference.py` is the separate instructor solution.
The fixed arrays `q_star`, `K_star`, and `V_star` match the chapter's worked example;
`q`, `K`, `V` are local aliases. The score symbols `s` remain independent:
construct `a` from them and compute `a.jacobian(s)` before fixture substitution.
The general proofs are in the chapter and author solutions, not inferred from
the three-source symbolic computations.

Tested environment: Python 3.12.3, SymPy 1.13.3. The pinned requirement is
`sympy==1.13.3`. No NumPy, GPU framework, model or dataset is required.

From the companion repository root, in an environment with this requirement:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_softmax_attention/student_setup.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_softmax_attention/reference.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s chapter_softmax_attention -p 'test_*.py' -v
```

The author entry point runs the same reference, avoiding a second copy of the
fixture. The reference calculates exact forward/masked/causal outputs, symbolic
Jacobians/Hessians and the constrained KL gap, independent loss derivatives,
50-digit central differences at three exact steps, shift identities, and both
deliberately incorrect implementations with their repairs. It checks invalid
support/temperature rejection, duplicate values, two heads, and temperature
limits. A direct symbolic directional derivative checks all three shared-X
branches against the matrix backward pass. The four behavioral tests check
domain rejection, causal exclusion, support order, and shared-input accumulation.
Computed expressions are compared to assertion targets.

Book source j is Python index j−1. An explicit allowed-index list defines the
support; symbolic −∞ is never differentiated. All local derivative checks hold
other local inputs independent. Floating-point output is labeled separately from
exact SymPy output; math.exp overflow is a recorded backend observation.

Run your script or notebook in a fresh process. Follow the chapter's
instructions for reporting assumptions, residuals, precision and errors.

Part B also checks the learned position scale with independent symbolic scores:
the fixed-support entropy derivative, all scale signs, equal scores, sparse
support, baseline initialization and the three equal-distractor limit regimes.
The exact compensation values are `4/7` and `16/31`.
