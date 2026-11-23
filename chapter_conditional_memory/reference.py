"""Exact Chapter 13 laboratory; constructed mathematics, not a model benchmark.

Run from the repository root:
    python3 code/chapter_conditional_memory/reference.py

Token codes are Python integers 0..3; BOS=4 is address padding only.
Vectors are columns, rows are zero-based, and parameters are fixed during a
trace. General fusion identities assume finite real inputs and parameters.
"""
from __future__ import annotations

from itertools import combinations, product
import sys

import sympy as sp

BOS, EOS = 4, 3
CORE = (0, 1, 3)
REPLAY = (0, 1, 0, 1, 2, 3)
T1 = sp.Matrix([0, 2, 1, -1, 3])
T2 = sp.Matrix([1, 0, -1, 2, 1, -1, 3])
E1 = sp.Matrix([[1, 0], [0, 1], [1, 1], [1, -1]])
E2 = sp.Matrix([[0, 1], [1, 0], [-1, 1], [2, 0]])
H_STAR = sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 6)])


def positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive Python integer")
    return value


def token_code(value: int) -> int:
    if type(value) is not int or value not in range(4):
        raise ValueError("Token codes must be Python integers in 0..3")
    return value


def sequence(tokens) -> tuple[int, ...]:
    tokens = tuple(tokens)
    if not tokens:
        raise ValueError("Supply one nonempty sequence")
    for token in tokens:
        token_code(token)
    if EOS in tokens[:-1]:
        raise ValueError("EOS may occur only at the end of one sequence")
    return tokens


def suffixes(tokens, order: int = 2) -> list[tuple[int, ...]]:
    """Batch padded suffixes; accepts a finished sequence or unfinished prefix."""
    tokens = sequence(tokens)
    positive_integer(order, "order")
    result = []
    for t in range(1, len(tokens) + 1):
        observed = tokens[max(0, t - order):t]
        result.append((BOS,) * (order - len(observed)) + observed)
    return result


class SuffixStream:
    """Address history only; convolution refinement would need additional state."""

    def __init__(self, maximum_order: int = 2):
        self.maximum_order = positive_integer(maximum_order, "maximum_order")
        self.reset()

    def reset(self) -> None:
        self.history: tuple[int, ...] = ()
        self.ended = False

    def push(self, token: int, order: int | None = None) -> tuple[int, ...]:
        token_code(token)
        n = self.maximum_order if order is None else positive_integer(order, "order")
        if n > self.maximum_order:
            raise ValueError("Requested order exceeds retained history")
        if self.ended:
            raise ValueError("Reset the stream before beginning another sequence")
        seen = self.history + (token,)
        key = (BOS,) * max(0, n - len(seen)) + seen[-n:]
        keep = self.maximum_order - 1
        self.history = seen[-keep:] if keep else ()
        self.ended = token == EOS
        return key


def address(key: tuple[int, int]) -> tuple[int, int]:
    """Complete padded code domain C^2; no random-family claim is made."""
    key = tuple(key)
    if len(key) != 2 or any(type(x) is not int or x not in range(5) for x in key):
        raise ValueError("A memory bigram has two integer codes in 0..4")
    a, b = key
    return (a + 2 * b) % 5, (2 * a + b) % 7


def read(key: tuple[int, int]) -> sp.Matrix:
    i, j = address(key)
    return sp.Matrix([T1[i], T2[j]])


def selection(addresses, capacity: int) -> sp.Matrix:
    positive_integer(capacity, "capacity")
    addresses = tuple(addresses)
    result = sp.zeros(len(addresses), capacity)
    for occurrence, row in enumerate(addresses):
        if type(row) is not int or not 0 <= row < capacity:
            raise ValueError("Row address outside the selected table")
        result[occurrence, row] = 1
    return result


def fusion(h, r, U, a, c, b=0) -> tuple[sp.Expr, sp.Matrix]:
    h, r, U, a, c = map(sp.Matrix, (h, r, U, a, c))
    if h.cols != 1 or r.cols != 1 or not h.rows or not r.rows:
        raise ValueError("h and r must be nonempty column vectors")
    if a.shape != h.shape or c.shape != r.shape or U.shape != (h.rows, r.rows):
        raise ValueError("Fusion shapes do not agree")
    alpha = 1 / (1 + sp.exp(-(a.dot(h) + c.dot(r) + sp.sympify(b))))
    return alpha, h + alpha * U * r


def zero_matrix(value) -> bool:
    value = sp.Matrix(value)
    return value.applyfunc(sp.simplify) == sp.zeros(*value.shape)


def exact_collision_experiment() -> dict:
    """Exhaust all 5^4 independent assignments of four labeled keys.

    This probability space is separate from the deterministic address().
    """
    pair_total = occupied_total = cases = 0
    for assigned in product(range(5), repeat=4):
        cases += 1
        pair_total += sum(assigned[i] == assigned[j] for i, j in combinations(range(4), 2))
        occupied_total += len(set(assigned))
    return {
        "cases": cases,
        "pair_total": pair_total,
        "occupied_total": occupied_total,
        "pairs": sp.Rational(pair_total, cases),
        "occupied": sp.Rational(occupied_total, cases),
        "excess": 4 - sp.Rational(occupied_total, cases),
    }


def payload(tokens, bytes_per_scalar: int = 2) -> dict:
    positive_integer(bytes_per_scalar, "bytes_per_scalar")
    signatures = [address(key) for key in suffixes(tokens)]
    requests = [(head, row) for signature in signatures for head, row in enumerate(signature)]
    distinct = set(requests)
    return {
        "stored_scalars": T1.rows + T2.rows,
        "stored_bytes": (T1.rows + T2.rows) * bytes_per_scalar,
        "logical_scalars": len(requests),
        "logical_bytes": len(requests) * bytes_per_scalar,
        "distinct_rows": len(distinct),
        "ideal_distinct_bytes": len(distinct) * bytes_per_scalar,
        "requests": requests,
        "distinct_requests": distinct,
    }


def table_storage(capacities, widths, element_bytes) -> int:
    """Separate physical tables; each counted once. All overhead excluded."""
    capacities, widths, element_bytes = map(tuple, (capacities, widths, element_bytes))
    if not capacities or not len(capacities) == len(widths) == len(element_bytes):
        raise ValueError("Table ledgers must have equal nonzero lengths")
    for values, name in ((capacities, "capacity"), (widths, "width"), (element_bytes, "element_bytes")):
        for value in values:
            positive_integer(value, name)
    return sum(B * width * size for B, width, size in zip(capacities, widths, element_bytes))


def ideal_overlap(payload_bytes, bandwidth_bytes_per_second, startup_seconds, lead_seconds):
    D, beta, latency, lead = map(sp.sympify, (
        payload_bytes, bandwidth_bytes_per_second, startup_seconds, lead_seconds
    ))
    if beta.is_positive is not True or any(x.is_nonnegative is not True for x in (D, latency, lead)):
        raise ValueError("Require positive bandwidth and nonnegative payload/times")
    transfer = latency + D / beta
    return transfer, sp.Max(0, transfer - lead)


def verify() -> dict:
    """Completed homework and numeric instructor checks, independently derived."""
    checks = 0

    def check(condition, name):
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    # A: direct indexing, linear selection, streaming and causality.
    expected_core = [(4, 1), (2, 1), (2, 5)]
    expected_replay = [(4, 1), (2, 1), (1, 2), (2, 1), (0, 4), (3, 0)]
    expected_reads = [sp.Matrix(v) for v in ((3, 0), (1, 0), (1, -1))]
    replay_reads = [sp.Matrix(v) for v in ((3, 0), (1, 0), (2, -1), (1, 0), (0, 1), (-1, 1))]
    for tokens, signatures, reads in ((CORE, expected_core, expected_reads), (REPLAY, expected_replay, replay_reads)):
        keys = suffixes(tokens)
        check([address(key) for key in keys] == signatures, "signature trace")
        check([read(key) for key in keys] == reads, "direct read trace")
        matrices = [selection([sig[k] for sig in signatures], table.rows) for k, table in enumerate((T1, T2))]
        gathered = (matrices[0] * T1).row_join(matrices[1] * T2)
        check(gathered == sp.Matrix([list(v) for v in reads]), "gather agrees with direct rows")
        stream = SuffixStream()
        check([stream.push(x) for x in tokens] == keys, "stream agrees with batch")
    check([address(g) for g in suffixes((0, 1, 2, 3))][:2]
          == [address(g) for g in suffixes(CORE)][:2], "future suffix does not alter prefix")
    stream = SuffixStream()
    for x in CORE:
        stream.push(x)
    stream.reset()
    check(stream.push(1) == (BOS, 1), "fresh sequence restores BOS")
    configurations = 0
    # Exhaust the stated finite domain, not every possible prefix/implementation.
    for length in range(1, 5):
        for previous in product(range(3), repeat=length - 1):
            for last in range(4):
                tokens = previous + (last,)
                for maximum_order in range(1, 5):
                    for requested_order in range(1, maximum_order + 1):
                        stream = SuffixStream(maximum_order)
                        check([stream.push(x, requested_order) for x in tokens]
                              == suffixes(tokens, requested_order), "finite streaming domain")
                        check(len(stream.history) <= maximum_order - 1, "stream history bound")
                        configurations += 1

    # B: differentiate independent symbols before substituting any fixture.
    h = sp.Matrix(sp.symbols("h0:2", real=True))
    r = sp.Matrix(sp.symbols("r0:2", real=True))
    a = sp.Matrix(sp.symbols("a0:2", real=True))
    c = sp.Matrix(sp.symbols("c0:2", real=True))
    U = sp.Matrix(2, 2, sp.symbols("u0:4", real=True))
    b = sp.symbols("b", real=True)
    alpha, y = fusion(h, r, U, a, c, b)
    J_r = y.jacobian(r)
    J_h = y.jacobian(h)
    check(zero_matrix(J_r - (alpha * U + alpha * (1 - alpha) * (U * r) * c.T)), "memory Jacobian identity")
    check(zero_matrix(J_h - (sp.eye(2) + alpha * (1 - alpha) * (U * r) * a.T)), "context Jacobian identity")
    zeros = sp.zeros(2, 1)
    r_star = sp.Matrix([1, -1])
    gate, anchor = fusion(H_STAR, r_star, sp.eye(2), zeros, zeros)
    check(gate == sp.Rational(1, 2), "constant gate")
    check(anchor == sp.Matrix([sp.Rational(7, 6), sp.Rational(1, 3)]), "exact interface output")
    context_gates = []
    for hidden, expected_gate, expected_y in (
        (zeros, sp.Rational(1, 2), sp.Matrix([sp.Rational(1, 2), -sp.Rational(1, 2)])),
        (sp.Matrix([1, 0]), sp.Rational(3, 4), sp.Matrix([sp.Rational(7, 4), -sp.Rational(3, 4)])),
    ):
        gate, output = fusion(hidden, r_star, sp.eye(2), sp.Matrix([sp.log(3), 0]), zeros)
        check(sp.simplify(gate - expected_gate) == 0, "context gate")
        check(zero_matrix(output - expected_y), "context output")
        context_gates.append(expected_gate)
    witness_values = list(zeros) + [1, 0] + list(zeros) + [1, 0] + list(sp.eye(2)) + [-1]
    witness = dict(zip(list(h) + list(r) + list(a) + list(c) + list(U) + [b], witness_values))
    missing = (J_r - alpha * U).subs(witness)
    check(missing == sp.diag(sp.Rational(1, 4), 0), "nonzero missing gate path")

    # C: two independent constructions of the table gradient.
    S = selection([sig[0] for sig in expected_core], 5)
    table_symbols = sp.Matrix(sp.symbols("t0:5", real=True))
    delta = sp.Matrix([2, 3, -1])
    diagnostic = (delta.T * S * table_symbols)[0]
    direct_gradient = sp.Matrix([sp.diff(diagnostic, x) for x in table_symbols])
    check(direct_gradient == S.T * delta, "symbolic adjoint")
    check(direct_gradient == sp.Matrix([0, 0, 2, 0, 2]), "repeated row accumulation")
    regularized = diagnostic + table_symbols.dot(table_symbols) / 2
    check(sp.Matrix([sp.diff(regularized, x) for x in table_symbols])
          == S.T * delta + table_symbols, "direct regularizer adds derivative")
    w = sp.symbols("w", real=True)
    losses = ((w - 1) ** 2 / 2, (w + 1) ** 2 / 2)
    collision_loss = sum(losses)
    check(sp.expand(collision_loss) == w ** 2 + 1, "conflicting quadratic")
    check(sp.solve(sp.diff(collision_loss, w), w) == [0], "stationary point")
    check(sp.diff(collision_loss, w, 2) == 2, "strict convexity")
    check(collision_loss.subs(w, 0) == 1, "minimum")
    check(tuple(sp.diff(loss, w).subs(w, 0) for loss in losses) == (-1, 1), "opposing contributions")
    check(collision_loss.subs(w, -1) == 2, "larger original local loss")
    predicts_positive = lambda value: value >= 0
    check(not predicts_positive(-1) and predicts_positive(0), "lower loss can worsen negative-case task")

    # D: deterministic witnesses and a separate exhaustive probability space.
    check(address((0, 1))[0] == address((1, 3))[0], "partial head-one collision")
    check(address((0, 1))[1] != address((1, 3))[1], "partial signature differs")
    check(read((0, 1)) != read((1, 3)), "partial collision does not equate reads")
    check(address((1, 0)) == address((BOS, 1)) == (1, 2), "legal full collision")
    check(read((1, 0)) == read((BOS, 1)) == sp.Matrix([2, -1]), "full collision retrieval")
    check(suffixes((1, 0))[-1] == (1, 0) and suffixes((1,))[0] == (BOS, 1), "collision keys admissible")
    check(address((0, 0)) != address((1, 2)) and read((0, 0)) == read((1, 2)), "equal learned rows refute converse")
    alt_code = lambda x: 0 if x == 2 else x
    check(tuple(map(alt_code, (0, 1))) == tuple(map(alt_code, (2, 1))), "isolated normalization alias")
    left, right = sp.Matrix([1, 0]), sp.Matrix([0, 1])
    check(left != right and sum(left) == sum(right) == 1, "concatenation versus sum")
    check(sp.ones(1, 2) * (left - right) == sp.zeros(1, 1), "projection can lose distinction")
    experiment = exact_collision_experiment()
    check(experiment["cases"] == 625, "complete assignment cardinality")
    check(experiment["pairs"] == sp.binomial(4, 2) / 5 == sp.Rational(6, 5), "exact pair mean")
    occupied_formula = 5 * (1 - sp.Rational(4, 5) ** 4)
    check(experiment["occupied"] == occupied_formula == sp.Rational(369, 125), "exact occupancy mean")
    check(experiment["excess"] == sp.Rational(131, 125) != experiment["pairs"], "excess differs from pairs")
    check(sp.binomial(4, 2) / (5 * 7) == sp.Rational(6, 35), "ideal independent full pairs")
    check(sp.binomial(4, 2) == 6 and 4 - 1 == 3, "one-bucket counterexample")

    # E: independent layout and byte/time accounting.
    selector_b = sp.Matrix([0, 1, 0, 0])
    packed = E1.row_join(E2)
    check(E1.T * selector_b == sp.Matrix([0, 1]), "layer-one B")
    check(E2.T * selector_b == sp.Matrix([1, 0]), "layer-two B")
    check(packed.shape == (4, 4) and len(packed) == len(E1) + len(E2) == 16, "packing count")
    for j in range(4):
        check(packed[j, :2] == E1[j, :], "all first-layer slices")
        check(packed[j, 2:] == E2[j, :], "all second-layer slices")
    check(2 * len(sp.zeros(2, 2)) == 8, "separate projection count")
    core_payload, replay_payload = payload(CORE), payload(REPLAY)
    check(tuple(core_payload[x] for x in ("stored_bytes", "logical_bytes", "ideal_distinct_bytes")) == (24, 12, 8), "core bytes")
    check(tuple(replay_payload[x] for x in ("stored_bytes", "logical_bytes", "ideal_distinct_bytes")) == (24, 24, 18), "replay bytes")
    check(core_payload["distinct_rows"] == 4 and replay_payload["distinct_rows"] == 9, "distinct identity rows")
    check(len({row for _, row in replay_payload["requests"]}) < replay_payload["distinct_rows"], "wrong cross-head deduplication loses rows")
    check(table_storage((5, 7), (1, 1), (2, 2)) == 24, "initial table storage")
    check(table_storage((50, 70), (1, 1), (2, 2)) == 240, "capacity-only storage expansion")
    check(table_storage((5, 7), (1, 1), (1, 2)) == 19, "mixed element sizes")
    check(15 + len(sp.zeros(2, 2)) + 2 + 2 + 1 == 24, "fixed full branch budget")
    transfer, exposed = ideal_overlap(4096, 10**9, sp.Rational(2, 10**6), sp.Rational(4, 10**6))
    check(transfer * 10**6 == sp.Rational(762, 125), "transfer microseconds")
    check(exposed * 10**6 == sp.Rational(262, 125), "exposed microseconds")
    check(ideal_overlap(4096, 10**9, 0, 1)[1] == 0, "enough lead hides modeled transfer")
    check(ideal_overlap(4096, 10**9, 0, 0)[1] == sp.Rational(4096, 10**9), "zero lead exposes modeled payload")
    return {
        "checks": checks, "stream_configurations": configurations,
        "collision_experiment": experiment,
        "core_bytes": {k: core_payload[k] for k in ("stored_bytes", "logical_bytes", "ideal_distinct_bytes")},
        "replay_bytes": {k: replay_payload[k] for k in ("stored_bytes", "logical_bytes", "ideal_distinct_bytes")},
        "transfer_us": transfer * 10**6, "exposed_us": exposed * 10**6,
    }


if __name__ == "__main__":
    print("Python:", sys.version.split()[0], "SymPy:", sp.__version__)
    result = verify()
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Exact mathematics and finite tests passed; no device or quality benchmark was run.")
