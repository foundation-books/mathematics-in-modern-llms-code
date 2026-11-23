"""Exact reference laboratory for Chapter 1 (author solution, not starter).

Run from the repository root: python3 code/chapter_language_modeling/ch01_examples.py
Only the standard library and SymPy are required. No floating-point input is
accepted by distribution validation. Unknown assumptions require a declaration.
"""

from dataclasses import dataclass
from itertools import product
import platform

import sympy as sp

R = sp.Rational
VOCABULARY = ("A", "B", "C", "<eos>")
OBSERVED = ("A", "B", "<eos>")
ALL_TARGETS = (1, 1, 1)
COMPLETION_TARGETS = (0, 1, 1)
ROWS = {
    (): (R(1, 2), R(1, 4), R(1, 8), R(1, 8)),
    ("A",): (R(1, 8), R(1, 4), R(1, 2), R(1, 8)),
    ("A", "B"): (R(1, 8), R(1, 8), R(1, 4), R(1, 2)),
}
DEFAULT_ROW = (R(1, 4),) * 4
CODEBOOKS = {
    (): ("0", "10", "110", "111"),
    ("A",): ("110", "10", "0", "111"),
    ("A", "B"): ("110", "111", "10", "0"),
}
DEFAULT_CODES = ("00", "01", "10", "11")
TIE_ORDER = ("<eos>", "A", "B", "C")
TOY_LOGITS = (sp.log(4), sp.log(2), sp.S.Zero, sp.S.Zero)


def equivalent(lhs, rhs):
    """Selected positive/real-log identities; no forced transformations."""
    return sp.simplify(sp.expand_log(lhs - rhs)) == 0


def validate_distribution(values):
    """Require exact, known finite real nonnegative entries and total one."""
    values = tuple(sp.sympify(x) for x in values)
    if not values or any(x.has(sp.Float) for x in values):
        raise ValueError("Use a nonempty exact distribution, not floats")
    if any(x.is_real is not True or x.is_finite is not True
           or x.is_nonnegative is not True for x in values):
        raise ValueError("Declare finite real nonnegative probabilities")
    if sp.simplify(sum(values) - 1) != 0:
        raise ValueError("Probabilities must sum to one")
    return values


def validate_prefix(prefix):
    prefix = tuple(prefix)
    if any(v not in VOCABULARY[:-1] for v in prefix):
        raise ValueError("An unfinished prefix contains only A, B, C")
    return prefix


def validate_complete(sequence):
    sequence = tuple(sequence)
    if not sequence or sequence[-1] != "<eos>":
        raise ValueError("A complete sequence ends in EOS")
    validate_prefix(sequence[:-1])
    return sequence


def conditional(prefix):
    return ROWS.get(validate_prefix(prefix), DEFAULT_ROW)


def codebook(prefix):
    return CODEBOOKS.get(validate_prefix(prefix), DEFAULT_CODES)


def prefix_probability(prefix):
    prefix = validate_prefix(prefix)
    return sp.prod(conditional(prefix[:t])[VOCABULARY.index(v)]
                   for t, v in enumerate(prefix))


def finite_reals(values):
    values = tuple(sp.sympify(x) for x in values)
    if not values or any(x.is_real is not True or x.is_finite is not True
                         for x in values):
        raise ValueError("Logits must be a nonempty finite real vector")
    return values


def softmax(logits):
    weights = tuple(sp.exp(x) for x in finite_reals(logits))
    return tuple(x / sum(weights) for x in weights)


def logsumexp(logits):
    return sp.log(sum(sp.exp(x) for x in finite_reals(logits)))


@dataclass(frozen=True)
class Score:
    selected: tuple
    probabilities: tuple
    probability_product: sp.Expr
    total_nats: sp.Expr
    mean_nats: sp.Expr
    total_bits: sp.Expr
    mean_bits: sp.Expr
    perplexity: sp.Expr


def score(sequence=OBSERVED, mask=ALL_TARGETS):
    """Score observed prefixes; the mask changes targets, never context.

    The selected product is a completion likelihood for the completion mask;
    an arbitrary mask's product need not be a completed-sequence probability.
    """
    sequence = validate_complete(sequence)
    mask = tuple(mask)
    if len(mask) != len(sequence) or any(m not in (0, 1) for m in mask):
        raise ValueError("Use one binary mask value per target")
    selected = tuple(t for t, m in enumerate(mask) if m == 1)
    if not selected:
        raise ValueError("At least one target must be selected")
    # Unselected losses are never evaluated; all prefixes still come from data.
    probabilities = tuple(conditional(sequence[:t])[VOCABULARY.index(sequence[t])]
                          for t in selected)
    total = (sp.oo if any(p == 0 for p in probabilities)
             else sp.expand_log(-sum(sp.log(p) for p in probabilities)))
    mean = total / len(selected)
    bits = sp.oo if total == sp.oo else sp.simplify(total / sp.log(2))
    return Score(selected, probabilities, sp.prod(probabilities), total, mean,
                 bits, bits / len(selected), sp.simplify(sp.exp(mean)))


def log_base(base):
    base = sp.sympify(base)
    if base.is_real is not True or base.is_finite is not True\
            or (base - 1).is_positive is not True:
        raise ValueError("The logarithm base must be known finite real > 1")
    return sp.log(base)


def known_support(values):
    values = validate_distribution(values)
    if any(x.is_zero is not True and x.is_positive is not True for x in values):
        raise ValueError("Support is unknown; use positive symbols or explicit zeros")
    return values


def entropy(q, base=sp.E):
    q = known_support(q)
    denominator = log_base(base)
    return sp.simplify(-sum(x * sp.log(x) for x in q if x != 0) / denominator)


def paired(q, p):
    q, p = known_support(q), known_support(p)
    if len(q) != len(p):
        raise ValueError("Use a common alphabet")
    return tuple(zip(q, p))


def cross_entropy(q, p, base=sp.E):
    terms, denominator = paired(q, p), log_base(base)
    if any(x != 0 and y == 0 for x, y in terms):
        return sp.oo
    return sp.simplify(-sum(x * sp.log(y) for x, y in terms if x != 0)
                       / denominator)


def kl(q, p, base=sp.E):
    terms, denominator = paired(q, p), log_base(base)
    if any(x != 0 and y == 0 for x, y in terms):
        return sp.oo
    return sp.simplify(sum(x * sp.log(x / y) for x, y in terms if x != 0)
                       / denominator)


def temperature(logits, tau):
    tau = sp.sympify(tau)
    if tau.is_positive is not True or tau.is_finite is not True:
        raise ValueError("Temperature must be known positive and finite")
    return softmax(tuple(x / tau for x in finite_reals(logits)))


def greedy(probabilities, vocabulary=VOCABULARY, tie_order=TIE_ORDER):
    """For exact concrete distributions; unknown symbolic ordering is rejected."""
    probabilities = validate_distribution(probabilities)
    if len(vocabulary) != len(probabilities) or set(tie_order) != set(vocabulary):
        raise ValueError("Vocabulary and tie order must match the distribution")
    largest = sp.Max(*probabilities)
    winners = [v for v, p in zip(vocabulary, probabilities) if equivalent(p, largest)]
    if not winners:
        raise ValueError("The ordering is unknown; instantiate symbolic weights")
    return next(v for v in tie_order if v in winners)


def two_step_paths():
    first = {"A": R(3, 5), "B": R(2, 5)}
    second = {"A": {"A": R(51, 100), "B": R(49, 100)},
              "B": {"A": R(99, 100), "B": R(1, 100)}}
    return {a + b: first[a] * second[a][b] for a, b in product("AB", repeat=2)}


def encode(sequence):
    sequence = validate_complete(sequence)
    return "".join(codebook(sequence[:t])[VOCABULARY.index(v)]
                   for t, v in enumerate(sequence))


def decode(bits):
    if not isinstance(bits, str) or not bits or any(b not in "01" for b in bits):
        raise ValueError("Use a nonempty binary string")
    prefix, offset = (), 0
    while offset < len(bits):
        codes = codebook(prefix)
        matches = [(v, c) for v, c in zip(VOCABULARY, codes)
                   if bits.startswith(c, offset)]
        if len(matches) != 1:
            raise ValueError("Invalid or truncated codeword")
        v, c = matches[0]
        offset += len(c)
        if v == "<eos>":
            if offset != len(bits):
                raise ValueError("Trailing bits after EOS")
            return prefix + (v,)
        prefix += (v,)
    raise ValueError("Code ended without EOS")


def rejection(call):
    """Assertions actually execute each deliberately invalid branch."""
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def run_checks():
    for row in (*ROWS.values(), DEFAULT_ROW):
        validate_distribution(row)
        assert all(x.is_positive is True for x in row)
    assert conditional(("C", "A")) == DEFAULT_ROW
    assert prefix_probability(("A", "B")) == R(1, 8)
    full, completion = score(), score(mask=COMPLETION_TARGETS)
    assert full.probability_product == R(1, 16)
    assert full.probabilities == (R(1, 2), R(1, 4), R(1, 2))
    assert equivalent(full.total_nats, 4 * sp.log(2))
    assert equivalent(full.mean_nats, R(4, 3) * sp.log(2))
    assert equivalent(full.perplexity, 2 ** R(4, 3))
    assert completion.probability_product == R(1, 8)
    assert completion.selected == (1, 2)
    assert equivalent(completion.mean_nats, R(3, 2) * sp.log(2))
    assert equivalent(completion.perplexity, 2 ** R(3, 2))
    for result in (full, completion):
        assert equivalent(result.total_nats, -sp.log(result.probability_product))
        assert equivalent(result.perplexity, 2 ** result.mean_bits)
    assert full.total_bits == 4 and completion.total_bits == 3

    z = sp.symbols("z0:4", real=True)
    c = sp.symbols("c", real=True)
    probabilities = softmax(z)
    assert equivalent(sum(probabilities), 1)
    assert all(equivalent(a, b) for a, b in zip(probabilities, softmax([x + c for x in z])))
    assert all(equivalent(-sp.log(p), logsumexp(z) - value)
               for p, value in zip(probabilities, z))
    substitution = dict(zip(z, TOY_LOGITS))
    assert tuple(sp.simplify(p.subs(substitution)) for p in probabilities) == ROWS[()]
    assert equivalent(logsumexp(TOY_LOGITS) - TOY_LOGITS[0], sp.log(2))

    q, u = ROWS[()], DEFAULT_ROW
    assert entropy(q, 2) == R(7, 4)
    assert cross_entropy(q, u, 2) == 2 and kl(q, u, 2) == R(1, 4)
    a = sp.symbols("a0:4", positive=True)
    b = sp.symbols("b0:4", positive=True)
    qs, ps = tuple(x / sum(a) for x in a), tuple(x / sum(b) for x in b)
    assert equivalent(cross_entropy(qs, ps), entropy(qs) + kl(qs, ps))
    assert equivalent(cross_entropy(q, u), entropy(q) + kl(q, u))
    binary_a, binary_b = (R(3, 4), R(1, 4)), (R(1, 2),) * 2
    ab, ba = kl(binary_a, binary_b), kl(binary_b, binary_a)
    assert equivalent(ab, R(3, 4) * sp.log(3) - sp.log(2))
    assert equivalent(ba, sp.log(2) - R(1, 2) * sp.log(3))
    assert equivalent(ab - ba, sp.log(R(243, 256)) / 4)
    assert ab != ba and sp.Rational(243, 256) != 1
    point = (sp.S.One, sp.S.Zero)
    assert entropy(point) == 0
    assert equivalent(cross_entropy(point, binary_b), sp.log(2))
    assert equivalent(kl(point, binary_b), sp.log(2))
    assert cross_entropy(point, point) == 0 and kl(point, point) == 0
    assert cross_entropy(binary_b, point) == sp.oo and kl(binary_b, point) == sp.oo

    low, high = temperature(TOY_LOGITS, R(1, 2)), temperature(TOY_LOGITS, 2)
    assert low == tuple(R(x, 22) for x in (16, 4, 1, 1))
    expected_high = tuple(x / (4 + sp.sqrt(2)) for x in (2, sp.sqrt(2), 1, 1))
    assert all(equivalent(x, y) for x, y in zip(high, expected_high))
    paths = two_step_paths()
    assert tuple(paths.values()) == tuple(R(x, 500) for x in (153, 147, 198, 2))
    assert sum(paths.values()) == 1
    first = greedy((R(3, 5), R(2, 5)), ("A", "B"), ("A", "B"))
    second = greedy((R(51, 100), R(49, 100)), ("A", "B"), ("A", "B"))
    assert first + second == "AA" and max(paths, key=paths.get) == "BA"
    assert greedy(conditional(("A",))) == "C"
    assert greedy(DEFAULT_ROW) == "<eos>"
    for prefix, codes in [*CODEBOOKS.items(), (("C",), DEFAULT_CODES)]:
        assert len(set(codes)) == 4
        assert all(not y.startswith(x) for i, x in enumerate(codes)
                   for j, y in enumerate(codes) if i != j)
        assert all(equivalent(len(c), -sp.log(p, 2))
                   for c, p in zip(codes, conditional(prefix)))
    assert encode(OBSERVED) == "0100" and decode("0100") == OBSERVED
    # Round trips visit the default rule and all canonical contexts, not only AB.
    for length in range(4):
        for prefix in product(VOCABULARY[:-1], repeat=length):
            seq = prefix + ("<eos>",)
            assert decode(encode(seq)) == seq

    invalid = [
        lambda: validate_distribution((R(1, 2), R(1, 3))),
        lambda: validate_distribution((2, -1)),
        lambda: validate_distribution((0.5, 0.5)),
        lambda: validate_distribution((sp.Symbol("w"), 1 - sp.Symbol("w"))),
        lambda: conditional(("D",)),
        lambda: conditional(("<bos>",)),
        lambda: conditional(("<eos>",)),
        lambda: score(("A", "B"), (1, 1)),
        lambda: score(("<eos>", "A", "<eos>")),
        lambda: score(("A", "D", "<eos>")),
        lambda: score(mask=(0, 0, 0)),
        lambda: score(mask=(1, 1)),
        lambda: score(mask=(1, 2, 1)),
        lambda: decode("0"),
        lambda: decode("1110"),
        lambda: decode("2"),
        lambda: temperature(TOY_LOGITS, 0),
        lambda: softmax((sp.oo, 0)),
        lambda: entropy(point, 1),
        lambda: cross_entropy(point, (1, 0, 0)),
        lambda: entropy((sp.Symbol("x", nonnegative=True)
                         / (1 + sp.Symbol("x", nonnegative=True)),
                         1 / (1 + sp.Symbol("x", nonnegative=True)))),
    ]
    for call in invalid:
        rejection(call)
    return {"invalid_cases": len(invalid), "round_trips": sum(3 ** n for n in range(4)),
            "full": full, "completion": completion,
            "binary_kl": (sp.expand_log(ab), sp.expand_log(ba)),
            "temperatures": (low, high), "paths": paths}


def main():
    results = run_checks()
    print("Python:", platform.python_version(), "SymPy:", sp.__version__)
    for name in ("full", "completion"):
        s = results[name]
        print(name, "product:", s.probability_product, "N:", len(s.selected),
              "total nats:", s.total_nats, "mean nats:", s.mean_nats,
              "bits:", s.total_bits, "PPL:", s.perplexity)
    print("H(q), CE(q,u), KL(q||u) in bits:", entropy(ROWS[()], 2),
          cross_entropy(ROWS[()], DEFAULT_ROW, 2), kl(ROWS[()], DEFAULT_ROW, 2))
    print("Binary KL (nats):", results["binary_kl"])
    print("Temperature 1/2 and 2:", results["temperatures"])
    print("Two-step paths:", results["paths"])
    print("Encode/decode:", encode(OBSERVED), decode("0100"))
    print("Exact identities and boundary checks passed; invalid cases rejected:",
          results["invalid_cases"], "; code round trips:", results["round_trips"])


if __name__ == "__main__":
    main()
