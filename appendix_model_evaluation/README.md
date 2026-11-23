# Appendix A: Exact Calculations for Model Evaluation

Optional exact calculations accompany **Mathematical Tools for Model
Evaluation**. The appendix prints the inputs, assumptions, and derivations.
The independent constructions are task mixtures, paired binary
outcomes with a supplied document grouping, factorial scores, and discrete
quality/cost configurations.

`homework_starter.py` preserves the earlier exact data and three tasks without answers.
`reference.py` derives the results; `test_reference.py` checks score reversal,
covariance signs, document aggregation, strict dominance, all utility crossings,
and exact-test size/symmetry for every orientation with 0–12 discordances.

Use the shared [companion setup](../README.md), then run from the companion repository root:

```sh
python3 check.py --appendix A
python3 appendix_model_evaluation/homework_starter.py
python3 appendix_model_evaluation/reference.py
python3 -m unittest discover -s appendix_model_evaluation -p 'test_*.py'
```

Validated on Python 3.12.3 and SymPy 1.13.3. Programs record actual versions;
do not use `-O`. No external dataset, model or training run is needed. The
sampling assumptions are mathematical models, not facts established by the
reference calculations. The optional calculations end with E's cap-three optimum.
