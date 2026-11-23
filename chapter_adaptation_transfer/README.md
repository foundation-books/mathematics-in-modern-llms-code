# Chapter 17: Post-training exact laboratory

The merged chapter has one four-part SymPy assignment. All inputs and written
proofs are in the chapter. The [shared companion guide](../README.md) describes
the pinned Python/SymPy environment and chapter selection.

Run from the companion repository root:

```sh
python3 chapter_adaptation_transfer/homework_starter.py
python3 chapter_adaptation_transfer/homework_reference.py
python3 -m unittest discover -s chapter_adaptation_transfer -p 'test_*.py'
python3 check.py --chapter 17
```

`homework_starter.py` supplies inputs and unsolved helper interfaces for A–D.
`homework_reference.py` is the complete current assignment reference:

- A differentiates the actual factor loss and checks the simultaneous update.
- B enumerates sequence laws and compares live and frozen visit derivatives.
- C constructs the regularized optimum and checks implicit scores and DPO.
- D completes both six-leaf trees, compares scoring rules and checks unseen mass.

All assertions use exact rationals and logarithms. The uniform-student readout
and the nonuniform preference reference are separate teaching cases. The six
response leaves form entire normalized laws; the observed pair is not
renormalized. Domains include 0<u,t<1, which positivity flags alone do not
encode. Finite checks supplement the general written proofs.

`reference.py` retains the earlier 73-check adaptation calculation set;
`preferences.py` retains the earlier preference helper/fixture set. These are
optional extended support, including temperature, both KL directions, teacher
aggregates, support limits, and comparison fitting. Their older A–F comments
record their origin and add no current homework requirements. The three test
modules cover the current integration and retained mathematical/domain behavior.

No checkpoint, dataset, GPU, network service or training experiment is used.
