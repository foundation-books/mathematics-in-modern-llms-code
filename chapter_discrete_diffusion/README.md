# Chapter 20 exact discrete diffusion laboratory

The printed Chapter 20 homework gives the assignment and all inputs.

For shared installation and chapter selection, see the
[companion guide](../README.md).

Python 3.10+ and SymPy **1.13.3**, the locally available version tested for this
implementation. The imported plan's earlier fixture run used 1.14.0; it is
historical evidence, not the dependency version of this executed solution.
No training, model downloads, datasets, GPU or network access are needed.

Run from the companion repository root:

```sh
python3 chapter_discrete_diffusion/homework_starter.py
python3 chapter_discrete_diffusion/reference.py
python3 -m unittest discover -s chapter_discrete_diffusion -p 'test_*.py' -v
```

`homework_starter.py` supplies conventions and data for A–D, including
lexicographic pair-state indexing. E and F are short applications. In E, report valid output tokens
per generation call (including commits) separately from tokens per
denoising evaluation, under the same counting rules at both costs.
`reference.py` is a complete author oracle, including exact 16-state joint
reversal and the synchronous product counterexample. Its JSON reports the
runtime versions and actual assertion count. Unsupported Bayes rows are
explicitly arbitrary normalized identity rows; their forward-data mass is zero.
Probability laws use rows; the inherited attention state and logit gradient
use columns. Pair states are ordered lexicographically.
Decimal evaluation occurs only in the entropy comparison. Toy latency units
are hypothetical costs, not measured Python or accelerator timings.
