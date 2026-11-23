"""Discrete diffusion student scaffold. Start with A–D; E and F are short applications."""

from itertools import product
import sympy as sp

R = sp.Rational
rho = sp.Matrix([[R(4, 15), R(1, 3), R(2, 5)]])
N_vocab = 3
I = sp.eye(N_vocab)
U = sp.ones(N_vocab, N_vocab) / N_vocab
betas = [R(1, 2), R(1, 2), sp.Integer(1)]
mask_index = N_vocab
pair_states = list(product(range(N_vocab + 1), repeat=2))
pair_index = {state: i for i, state in enumerate(pair_states)}
# Pair (i,j) has index 4*i+j. Product entry to (u,v): M[i,u]*M[j,v].
# Build rows yourself; do not confuse a conditional row with propagated mass.
# Example blank report: (time, current_pair, current_mass, reverse_row_sum).
assert all(pair_index[(i, j)] == 4*i+j for i, j in pair_states)

# A: Construct and compose both kernel families; check stochasticity.
# B: Write posterior(previous, step, clean, current); reject zero support.
# C: Derive symbolic loss, gradient, and corrected timestep estimator.
# D: Enumerate all 16 states; distinguish supported Bayes rows from extensions.
# E: Separate denoising evaluations from all generation calls (including commits).
# Compare valid tokens per denominator, positions and latency at both costs.
# F: Compute entropy, test the bounded rule, and interpret both failures.
# Use exact arithmetic until the explicitly numerical entropy comparison.
