# Chapter 10 SymPy laboratory

For shared installation and chapter selection, see the
[companion guide](../README.md).

This is a width-four mathematical teaching program, not an upstream model
implementation, training run or benchmark. It uses Python 3.10+ and SymPy;
the executed reference environment is Python 3.12.3, SymPy 1.13.3.
No model weights, dataset, accelerator or network access is required to run it.

Run from the companion repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 chapter_gates_normalization/homework_starter.py
PYTHONDONTWRITEBYTECODE=1 python3 chapter_gates_normalization/reference.py
```

`homework_starter.py` is printed in the final homework section and supplies all
essential data and helpers. Add your Tasks A–F below its final comment.
`chapter10_fixture.py` keeps the same exact fixture with explicit lift and
perturbation. `reference.py` is a separate instructor solution/checker with
independent symbolic differentiation, exact domain witnesses, four ranks,
two gate paths and three 60-digit central differences. It prints a compact
execution summary; `--json /tmp/chapter10-reference.json` also saves its record.

Helpers accept symbolic real inputs whose legal domains may not be decidable
automatically. State nonnegative epsilon and nonzero denominators yourself;
explicit zero-stabilizer zero/constant inputs and malformed shapes are rejected.
Failed simplification is not automatically a disproof. The script checks
dimension-four identities and named examples; general-width proofs are in the
chapter. Values, local derivatives, relative coordinate dominance and deep
training are different conclusions.
