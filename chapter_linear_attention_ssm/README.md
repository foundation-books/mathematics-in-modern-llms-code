# Chapter 8 SymPy homework companion

For shared installation and chapter selection, see the
[companion guide](../README.md).

One integrated homework, *One sequence, four mathematical views*. The complete
student assignment is in Section 8.11 of the book. `student_setup.py` provides
exact fixtures, an equality utility, and student interfaces without solutions.
`reference.py` is separate instructor material; `test_reference.py` checks its
behavior and rejection contracts. The chapter gives the general proofs.

Checked environment: Python 3.12.3, SymPy 1.13.3. The dependency is pinned in
`requirements.txt`. No other library, model, network service or GPU is required.
From the companion repository root, using an environment with that dependency:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_linear_attention_ssm/student_setup.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_linear_attention_ssm/reference.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s chapter_linear_attention_ssm -p 'test_*.py' -v
```

Public operators receive nonempty sequences of exact finite real SymPy column
matrices. Attention receives already transformed, elementwise nonnegative
query/key features and signed values. Each denominator must be provably
positive. Empty inputs, wrong shapes/lengths, floating-point fixture entries,
unsupported positivity assumptions, and invalid retention factors raise
`ValueError`. Saved states must come from the same feature stream; shapes and
normalizer nonnegativity are checked, but history provenance is a caller
contract. All incoming matrices are copied.

Pairwise and expanded routines return output lists and are independent of
the recurrent implementations. Attention recurrences return outputs and `(S,n)`
prefix states; the SSM recurrence returns outputs and state columns. The affine
prefix reference returns states and prefix pairs. `balanced_reduce` returns
one chronological segment pair. Repeating it over tiny prefixes in checks is
not a work-efficient parallel scan or a performance implementation.

The reference checks every attention/SSM prefix, rectangular states,
continuation, future exclusion, decay and zero gates, initial-state response,
direct feedthrough, ZOH and its singular limit, collision and rank witnesses,
noncommuting/singular affine composition, deliberate normalization mistakes,
and input errors. These are finite exact checks, not quality or speed evidence.

The normalizer vector $n_t$ is `normalizer_state`; its scalar query read is $Z_t$.
