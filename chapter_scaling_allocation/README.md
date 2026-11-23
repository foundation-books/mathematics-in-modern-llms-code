# Chapter 15 symbolic allocation laboratory

## Homework modules

Choose fitting and allocation (A, B, F1(a), F2), acquisition (C–E, F3),
or evaluation sampling (F1(b)). Acquisition inputs are shared across C–E;
F1(b) uses Appendix A's paired counts and document grouping independently
of F1(a)'s response-surface perturbation. The expansion starter supplies
those data without solving Part F. The expansion reference checks the
conditional means of the second corrected contribution and the two-round
average separately. For F1(b), it reconstructs both means, sample variances,
squared estimated standard errors, the iid-item conditional exact p-value,
and the threshold for ten prespecified valid comparisons. The two iid models
have different independent units; regrouping observations does not establish
independence. No document-level p-value or confidence interval is inferred.

Run all documented starters, references and tests with:

    python3 check.py --chapter 15 --appendix A

## Starter and references

Begin with [expansion_student.py](expansion_student.py) and the printed
homework. [reference.py](reference.py) checks the foundation calculations
and [expansion_reference.py](expansion_reference.py) checks the independent
extensions.

For shared installation and chapter selection, see the
[companion guide](../README.md).

These are deterministic constructed response surfaces, not measured LLM
training results. The manuscript uses `M_draw` for a positive integer IID draw
count and `n` for normalized model size; the expansion reference's local integer
`n` is only a draw-count symbol. General adaptive identities need the actual
past-conditioned proposal and finite first moment. The companion uses Python
3.10+ and SymPy. The chapter's official documentation
sources identify SymPy 1.14.0. No model, dataset, GPU, or new training is needed.

Run from the companion repository root:

```sh
python3 chapter_scaling_allocation/expansion_student.py
python3 chapter_scaling_allocation/reference.py
python3 chapter_scaling_allocation/expansion_reference.py
python3 check.py --chapter 15
```

`homework_student.py` preserves the earlier foundation scaffold, with inputs
and shape assertions. `reference.py`
implements the six tasks and verifies the supplied anchors, including
rank deficiency, caps, repetition endpoints, equal-quality lifetime costs,
and the specified asymmetric perturbation. It distinguishes exact symbolic
identities from finite witnesses. The chapter gives the general uniqueness
and allocation proofs.

Tests require pytest in addition to SymPy. `requirements.txt` lists runtime
dependencies; no dependency installation was performed during authoring.
All homework inputs, fitting weights, expected output types, and interpretation
requirements appear in the final chapter section. Instructor answers stay
outside the printed chapter.
