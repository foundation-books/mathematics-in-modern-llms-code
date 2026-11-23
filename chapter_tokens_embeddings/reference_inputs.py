"""Frozen exact inputs for Chapter 4; no model or data downloads."""

import sympy as sp

VOCABULARY = ("A", "B", "C", "<eos>", "D", "E", "F", "G")
DECODED_TEXT = ("a", "b", "c", "", "ab", "ba", "ac", "bc")
IDS = (0, 1, 0, 2)
POSITIONS = (0, 1, 2, 3)
E = sp.Matrix([
    [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
])
S = sp.Matrix(4, 8, lambda i, j: int(IDS[i] == j))
W = sp.Matrix([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
GX = sp.Matrix([[1, 2, 0, 0], [0, 1, 1, 0], [3, 0, 0, 1], [0, 0, 2, 1]])
GAMMA = sp.Matrix(4, 8, lambda i, j: int(j == i + 1))
H_EXPECTED = sp.Matrix([[1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [1, 2, 0, 0]])
GRADIENT_EXPECTED = sp.Matrix([
    [1, 1, 3, 2], [3, 2, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1],
    [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
])
OMEGA = (sp.pi / 2, sp.pi / 4)
FULL_SCORE = 1 + sp.sqrt(2) / 2
PARTIAL_SCORE = sp.sqrt(2) / 2
RESCALED_SCORE = sp.sqrt(2) / 2 + sp.sqrt(2 - sp.sqrt(2)) / 2
