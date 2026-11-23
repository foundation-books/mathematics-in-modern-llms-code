# Chapter 19 exact speculative-decoding laboratory

The printed homework offers a sampler route (A1–A6) and a constrained
language route (B–F) with separate fixtures. For A4 the foundation
reference checks first accepted-token probabilities, the reached second
rate, and survival and progress when only the proposal after C changes.
Its stationary grid is a separate cost calculation. Both routes use the
ordinary exactness proof. [enumeration_scaffold.py](enumeration_scaffold.py)
stores each incoming mass, conditional probability and resulting child mass
separately, including stopped failures. It supplies no target/proposal law.
Run it from the companion repository root with
`python3 chapter_speculative_decoding/enumeration_scaffold.py` and check
it with `python3 -m pytest -q chapter_speculative_decoding/test_enumeration_scaffold.py`.

For shared installation and chapter selection, see the
[companion guide](../README.md).

Requires Python 3.10+ and SymPy. Start with
[homework_starter.py](homework_starter.py) for the sampler route or use
[enumeration_scaffold.py](enumeration_scaffold.py) for branch bookkeeping.
The printed homework defines both routes.

Run from the companion repository root:

    python3 chapter_speculative_decoding/homework_starter.py
    python3 chapter_speculative_decoding/reference.py
    python3 check.py --chapter 19

`expansion_student.py` and `expansion_reference.py` supply optional
extensions from the earlier assignment.

The complete reference validates rational categorical laws, computes
acceptance/residual/coupling, enumerates every two-token branch, differentiates
entropy and weighted cross-entropy, evaluates the eight-horizon cost grid, and
compares complete EOS-stopped sequence laws with independent target products.
Behavioral tests cover mathematical outputs and invalid inputs.

The reference includes 225 rational-simplex law pairs,
nine draft pairs/21 positive branches, both biased algorithms, eight cost
choices, 12 teaching horizon/budget cases and 9 early/rejected-EOS cases.
The behavioral suite has 21 cases. Tested Python 3.12.3 / SymPy 1.13.3;
official mutable documentation inspected for exact-number conventions
identifies 1.14.0. No dependency was installed.

Finite checks support the general proofs and do not measure GPU performance
or certify production implementations. The stationary constructed cost
fixture is separate from the prefix-dependent chain.
