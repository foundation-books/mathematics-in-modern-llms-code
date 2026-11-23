# Sharing and compressing attention state companion

For shared installation and chapter selection, see the
[companion guide](../README.md).

Chapter 6's single homework, *One Attention Computation, Several State
Representations*, is fully specified in the printed book. `fixture.py` supplies
the exact teaching matrices and starter helpers. `reference.py` is the separate
instructor reference. Start with A–D and F; E1 (position equivalence)
and E2 (role sharing and storage) can be assigned separately.

Run from the companion repository root in an environment with SymPy 1.13.3:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_attention_state/fixture.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_attention_state/reference.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s chapter_attention_state -p 'test_*.py' -v
```

The reference checks expanded/absorbed logits, weights, head outputs and
combined outputs for all four heads and all three inclusive causal prefixes.
MHA/GQA/MQA are checked against each variant's own direct expansion. Additional
checks cover rank/fixed-projection loss, all24 legal positional scores,
partial-RoPE reconstruction, future-input derivatives and integer cache counts.
Symbolic equalities supplement the general proofs printed in the chapter.

Mathematical head/position/group numbers are one-based; Python indices and group
maps are zero-based. Exact SymPy expressions are not a simulation of the
illustrative two-byte storage ledger. The example runs on a CPU with the
shared dependencies.
