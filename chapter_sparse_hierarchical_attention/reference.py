"""Exact teaching operators for Chapter 7; no performance measurements.

Vectors are columns, K/V store rows, and all external IDs are one-based.
The mean-key router computes actual key averages and scaled query dot products.
"""

from functools import cmp_to_key
import sympy as sp


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def fixture(exponential_weights=(9, 1, 4, 4, 1, 1, 2, 2)):
    w_exp = tuple(sp.sympify(x) for x in exponential_weights)
    if not 1 <= len(w_exp) <= 8 or any(x.is_positive is not True or
                                  x.is_finite is not True for x in w_exp):
        raise ValueError("fixture weights must be finite and positive, with length 1..8")
    q = sp.Matrix([sp.sqrt(2), 0])
    K = sp.Matrix([[sp.log(x), 0] for x in w_exp])
    V = sp.Matrix([(1, 0), (0, 0), (0, 1), (0, 1),
                   (0, 0), (0, 0), (1, 1), (1, 1)][:len(w_exp)])
    Q = sp.Matrix([list(q) for _ in w_exp])
    return Q, K, V


def check_keys(t, q, K):
    integer(t, "t", 1)
    if K.rows < t or K.cols < 1 or q.shape != (K.cols, 1):
        raise ValueError("q must be a key-width column and K must contain t rows")
    if any(x.is_real is not True or x.is_finite is not True for x in (*q, *K)):
        raise ValueError("query and key entries must be finite real numbers")


def support_ids(t, support):
    ids = tuple(support)
    if not ids:
        raise ValueError("support must be nonempty")
    for j in ids:
        integer(j, "position", 1)
        if j > t:
            raise ValueError("support contains a future or out-of-range position")
    if len(set(ids)) != len(ids):
        raise ValueError("support contains duplicate positions")
    return tuple(sorted(ids))


def check_values(K, V):
    if V.rows != K.rows or V.cols < 1:
        raise ValueError("V must have the same row count as K and positive width")
    if any(x.is_real is not True or x.is_finite is not True for x in V):
        raise ValueError("values must be finite and real")


def restricted_attention(t, q, K, V, support):
    """Return a t-entry probability column and a value-width output column."""
    check_keys(t, q, K)
    check_values(K, V)
    ids = support_ids(t, support)
    exponential_weights = {j: sp.simplify(sp.exp((q.T * K.row(j - 1).T)[0]
                                    / sp.sqrt(K.cols))) for j in ids}
    denominator = sum(exponential_weights.values(), sp.S.Zero)
    a = sp.zeros(t, 1)
    for j, weight in exponential_weights.items():
        a[j - 1] = sp.simplify(weight / denominator)
    return a, (V[:t, :].T * a).applyfunc(sp.simplify)


def top_k(scores, k):
    """Select finite real scores by descending value, then immutable integer ID.

    Nontrivial budgets return ranking order; an all-candidate result uses ID
    order. Reject undecidable comparisons only when selection needs them.
    """
    integer(k, "k")
    clean = {}
    for j, score in scores.items():
        integer(j, "candidate ID", 1)
        value = sp.sympify(score)
        if value.is_real is not True or value.is_finite is not True:
            raise ValueError("scores must be finite and real")
        clean[j] = value

    if k == 0:
        return ()
    if k >= len(clean):
        return tuple(sorted(clean))

    def compare(i, j):
        gap = sp.simplify(clean[i] - clean[j])
        if gap.is_zero:
            return (i > j) - (i < j)
        if gap.is_positive:
            return -1
        if gap.is_negative:
            return 1
        raise ValueError("score ordering is undecidable; supply ordered inputs")

    return tuple(sorted(clean, key=cmp_to_key(compare))[:k])


def block_route(t, q, K, b=2, k_b=1, *, faulty_current=False):
    """Return historical/competing block IDs, token support, and routing scores.

    faulty_current=True is the intentional counterexample: a full current
    block competes for a routing slot before illegal output IDs are removed.
    """
    check_keys(t, q, K)
    integer(b, "b", 1)
    integer(k_b, "k_b")
    current = (t - 1) // b + 1
    scores = {}
    stop = current + 1 if faulty_current else current
    for u in range(1, stop):
        rows = range((u - 1) * b, min(u * b, K.rows))
        mean = sum((K.row(j).T for j in rows), sp.zeros(K.cols, 1)) / len(rows)
        scores[u] = sp.simplify((q.T * mean)[0] / sp.sqrt(K.cols))
    chosen = top_k(scores, k_b)
    support = set(range((current - 1) * b + 1, t + 1))
    for u in chosen:
        support.update(j for j in range((u - 1) * b + 1, u * b + 1) if j <= t)
    return chosen, tuple(sorted(support)), scores


def packed_route(requests, request_id, t, q, b=2, k_b=1):
    """requests maps identity to (padded K, padded V, real_length).

    Slice real positions before constructing summaries. Returns local IDs and
    output; request partitions are independent even if storage is packed.
    """
    K, V, real_length = requests[request_id]
    integer(real_length, "real_length", 1)
    if real_length > K.rows or V.rows != K.rows or t > real_length:
        raise ValueError("invalid real request boundary")
    real_K, real_V = K[:real_length, :], V[:real_length, :]
    chosen, support, scores = block_route(t, q, real_K, b, k_b)
    a, output = restricted_attention(t, q, real_K, real_V, support)
    return chosen, support, scores, a, output


def block_statistics(t, q, K, V, blocks):
    """Exact query-dependent Z and mu for a partition of the legal prefix."""
    check_keys(t, q, K)
    check_values(K, V)
    ids = [support_ids(t, block) for block in blocks]
    flat = tuple(j for block in ids for j in block)
    if sorted(flat) != list(range(1, t + 1)):
        raise ValueError("blocks must partition the real legal prefix")
    masses, means = [], []
    for block in ids:
        exponential_weights = [sp.simplify(sp.exp((q.T * K.row(j - 1).T)[0]
                                     / sp.sqrt(K.cols))) for j in block]
        mass = sum(exponential_weights, sp.S.Zero)
        weighted = sum((w * V.row(j - 1).T for j, w in zip(block, exponential_weights)),
                       sp.zeros(V.cols, 1))
        masses.append(mass)
        means.append((weighted / mass).applyfunc(sp.simplify))
    return tuple(masses), tuple(means)


def pooled_attention(q, K, V, historical_blocks, recent, *, correct_size=False):
    """Mean replacements plus distinct recent singletons in a supplied prefix.

    K/V must contain only that query's real legal prefix; slice before calling.
    This helper has no separate query-position argument.
    """
    check_keys(K.rows, q, K)
    check_values(K, V)
    blocks = [tuple(block) for block in historical_blocks]
    records = blocks + [(j,) for j in recent]
    flat = tuple(j for block in records for j in block)
    support_ids(K.rows, flat)
    keys, values, exponential_weights = [], [], []
    for block in records:
        if not block:
            raise ValueError("a summary block must be nonempty")
        keys.append(sum((K.row(j - 1).T for j in block),
                        sp.zeros(K.cols, 1)) / len(block))
        values.append(sum((V.row(j - 1).T for j in block),
                          sp.zeros(V.cols, 1)) / len(block))
        logit = (q.T * keys[-1])[0] / sp.sqrt(K.cols)
        if correct_size:
            logit += sp.log(len(block))
        exponential_weights.append(sp.simplify(sp.exp(logit)))
    output = sum((w * value for w, value in zip(exponential_weights, values)),
                 sp.zeros(V.cols, 1)) / sum(exponential_weights, sp.S.Zero)
    return tuple(exponential_weights), output.applyfunc(sp.simplify)


def pair_counts(T, b, k_b, w=2):
    integer(T, "T", 1)
    integer(b, "b", 1)
    integer(k_b, "k_b")
    integer(w, "w", 1)
    if T % b:
        raise ValueError("the exact count formula assumes T is divisible by b")
    n_b = T // b
    main = n_b * b * (b + 1) // 2 + b * b * sum(min(k_b, c - 1)
                                               for c in range(1, n_b + 1))
    route = b * n_b * (n_b - 1) // 2
    return T * (T + 1) // 2, sum(min(w, t) for t in range(1, T + 1)), main, route


def policy_supports():
    return {
        "dense": tuple(range(1, 9)),
        "local": (7, 8),
        "mean-key block": (3, 4, 7, 8),
        "block mass oracle": (1, 2, 7, 8),
        "token mass oracle": (1, 3, 7, 8),
    }
