# Chapter 13 exact conditional-memory laboratory

For shared installation and chapter selection, see the
[companion guide](../README.md).

Run from the companion repository root with Python 3.10 or later and SymPy:

```sh
python3 chapter_conditional_memory/homework_starter.py
python3 chapter_conditional_memory/reference.py
python3 -m unittest discover -s chapter_conditional_memory -p 'test_*.py' -v
```

The supplied environment has Python 3.12.3 and SymPy 1.13.3. Dependencies are
declared in `requirements.txt`; no framework, dataset, model weights, GPU,
network service, or training run is required.

`homework_starter.py` supplies the printed direct-read interface and all fixed
matrices, without the completed tasks. `reference.py` is the completed
author/instructor reference. `test_reference.py` tests legal boundaries,
invalid-input handling, atomic streaming rejection/reset, zero-history order
one, table-aware deduplication and agreement with the printed starter.

The reference derives both symbolic fusion Jacobians, a nonzero missing-path
witness, gather gradients in two ways, and conflicting-target minima. It
enumerates all 625 independent four-key/five-bucket assignments. A separate
finite streaming check covers every legal sequence of lengths one through
four (non-EOS codes before the final position), maximum orders one through
four, and each requested order not exceeding that maximum: 1600 configurations.
Neither finite domain establishes a theorem for every implementation; the
general proofs are in the chapter and instructor key.

Tokens are Python integer codes 0..3. BOS=4 is address padding, not a predicted
token. Every sequence resets history, EOS can be only its last token, and
table rows are zero-based. The hash is defined on all 25 padded-code bigrams,
including keys not realized by legal prefixes. Hash heads retain separate
table identities. The EOS diagnostic gradient is not a next-token loss.

All table and gate parameters are fixed during a read trace. Logical payload
uses two accounting bytes per scalar, not SymPy object storage. Ideal distinct
payload assumes an empty cache and perfect retention. Actual traffic, trained
quality, model-code parity and hardware latency are not measured. The random
assignment experiment does not establish independence of the fixed hashes.
