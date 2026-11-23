# Chapter 5 efficient attention laboratory

The chapter has eight written exercises with worked answers and one
four-part SymPy homework. The printed assignments give all inputs.
The starter supplies their exact values and unsolved summary interfaces.
The executable reference checks each part independently of the starter.

| Part | Calculation |
| --- | --- |
| A | Compare direct and online attention, then differentiate the normalized expression before substituting the inputs. Check the reconstructed gradients, the stale normalizer and the lost coupling term. |
| B | Align two partitions, check the averaging and unaligned-sum errors, and compare direct causal rows with clipped tiles. Change only the third value and track the original allowed pairs. |
| C | Distinguish logical and issued contractions, tile capacity, and memory traffic. The 128-byte tile's traffic count assumes enough fast memory for it. The rate calculation reports service-time lower bounds. |
| D | Prove the exact common-score shift and value-scale identities. Compare finite outputs with the independent reference and check both cancellation orders. |

Use Python 3.10+ and the dependencies in the
[companion guide](../README.md). Run from the companion repository root.

```sh
python3 chapter_efficient_execution/homework_starter.py
python3 chapter_efficient_execution/reference.py
python3 -m pytest -q chapter_efficient_execution/test_reference.py
python3 check.py --chapter 5
```

The tiled calculation forms scores for the active query and key blocks.
Code indices start at zero. The traversal records original allowed pair
identities and their multiplicities. The reference sweeps all six paired
record permutations, both binary trees per permutation, all four
contiguous chunkings per permutation, and the nine tile shapes in
`{1, 2, 4}^2`. Each shape covers all six allowed pairs exactly once.
It also checks the masked matrix gradients by differentiating the
three-position example.

For Part D, begin with [working_precision.py](working_precision.py), which
provides rounded primitives and an independent exact-weight reference, leaving
the attention algorithm to implement. Run
`python3 chapter_efficient_execution/working_precision.py`; it reports
`10000000000` at 8 digits and `10000000001` at 12. Each input conversion and
primitive operation rounds in a private context; Python floats are rejected.
Construct fractional inputs with `Fraction`, not an already rounded decimal.
Record both decimal digits and the reported binary precision. Run the focused
checks with `python3 -m pytest -q chapter_efficient_execution/test_working_precision.py`.

In the completed reference, mpmath forms operands and performs arithmetic at 8 and 12
decimal working digits. The reference starts from exact integer weights
and uses 70 decimal digits for the division. The criterion checks
finiteness and uses the maximum absolute output entry, with
`atol=1e-7` and `rtol=1e-5`. At 8 digits, adding the unit term before
the two large terms cancel can round it away. The log shows both
orders and their tolerance results.

The earlier `expansion_student.py`, `expansion_reference.py`,
`extensions_reference.py` and `inference_schedule.py` are historical
companions. The chapter checker runs the current starter and reference
and includes the preserved tests for those earlier examples.
