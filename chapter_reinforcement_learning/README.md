# Chapter 18 symbolic laboratory

## What the policy choices represent

The calculator law enumerates two abstract assistant decisions: choose
`DIRECT` or call the calculator (`VERIFY`), then choose final answer `5` or
`4`. An LLM expresses requests and answers through generated tokens; these
labels do not prescribe literal vocabulary entries or a separate controller.
A completed generated request has probability equal to the product of its
token conditionals, including generated ending decisions. The surrounding
program executes it and inserts the tool's observation before generation
resumes. The chapter's small sigmoid policy is separately specified for
exact calculation.

Here `p` is the call probability and `q` is the assistant's probability of
answering `5` after observing the calculator's deterministic `5`. The feature
is frozen, `(w, phi)` has three scalar trainable entries, and the direct
answer law stays fixed at equal halves, with zero trainable score. Its halves
still belong in complete path probabilities. The main cost is `1/10`; only
the rollout-group example uses cost zero, including all 256 ordered groups.

The auxiliary binary tree is a separate policy, interpreted as two token
positions for the clipping comparison. Independent reduction arrays and
fragment lengths count included generated-token contributions. A storage
boundary adds no decision and does not reset conditioning. Ownership, parameter
dependence, and loss inclusion remain separate questions.

## Starter and references

Complete foundation tasks A1–A8, then choose B (group estimators),
C–E (reward and update objectives), or F–G (collection and fragment weights).
Begin with [expansion_student.py](expansion_student.py).
[reference.py](reference.py) checks foundation calculations and
[expansion_reference.py](expansion_reference.py) checks selected extensions,
including C1’s capped length charges and G’s two-token/one-token split.

For shared installation and chapter selection, see the
[companion guide](../README.md).

This CPU-only laboratory verifies the constructed policy in Chapter 18.
It needs Python 3.10+ and SymPy; the implementation was checked with SymPy
1.13.3.

From the companion repository root:

```sh
python3 chapter_reinforcement_learning/expansion_student.py
python3 chapter_reinforcement_learning/reference.py
python3 chapter_reinforcement_learning/expansion_reference.py
python3 check.py --chapter 18
```

`homework_student.py` and `homework_student.ipynb` supply an optional
earlier six-task investigation with the same Python code. The reference
enumerates all 256 ordered groups for the finite expectation.

The JSON output separates named exact assertions from the final numerical
sigmoid evaluation. The finite-tree expected-gradient step does not measure
real LLM learning. Symbol domains include `0<p,q<1`; clipping kinks have no
unique derivative. Ownership and validity are different masks. Termination
zeroes a bootstrap; an administrative truncation preserves it.

The elective B/D/G extensions in `expansion_reference.py` additionally check
group centering and pooled scaling (`13/100`, magnitude ratio `5:1`), the
unequal-size variance identity, detached capped coefficients versus PPO for
both advantage signs, normalized binary derivatives and central differences
with the coefficient frozen. The two-group token, response and global-token
reductions are `1/4`, `1/2` and `0`. These are exact teaching constructions;
the population divisor and zero-variance rule do not assert an undisclosed
production stabilizer.
