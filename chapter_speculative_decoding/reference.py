"""Exact speculative decoding homework reference; no timing or pretrained model claims.

Run from the repository root: python3 code/chapter_speculative_decoding/reference.py
All finite probabilities are rational. Symbolic derivatives are checked separately.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json
import sys

import sympy as sp

R = sp.Rational
TOKENS = ("A", "B", "C", "EOS")
o_star = sp.Matrix([R(2, 3), R(5, 6)])
WT = sp.Matrix([[R(3, 2)*sp.log(3), 0], [0, R(6, 5)*sp.log(2)], [0, 0], [0, 0]])
WD = sp.Matrix([[R(3, 2)*sp.log(2), 0], [0, R(6, 5)*sp.log(3)], [0, 0], [0, 0]])
p = sp.Matrix([R(1, 2), R(1, 3), R(1, 6)])
q = sp.Matrix([R(1, 3), R(1, 2), R(1, 6)])
P = sp.Matrix([[R(1, 2), R(1, 4), R(1, 4)],
               [R(1, 4), R(1, 2), R(1, 4)],
               [R(1, 4), R(1, 4), R(1, 2)]])
Q = sp.Matrix([[R(1, 4), R(1, 2), R(1, 4)],
               [R(1, 2), R(1, 3), R(1, 6)],
               [R(1, 4), R(1, 4), R(1, 2)]])


def categorical(values):
    """A nonempty exact rational categorical column; do not coerce floats."""
    v = sp.Matrix(values)
    if v.rows == 0 or v.cols != 1:
        raise ValueError("Expected a nonempty probability column.")
    if any(x.is_Rational is not True or x < 0 for x in v):
        raise ValueError("Use exact nonnegative rational probabilities.")
    if sum(v) != 1:
        raise ValueError("Probabilities must sum to one exactly.")
    return v


def softmax(logits, temperature=sp.Integer(1)):
    temperature = sp.sympify(temperature)
    if temperature.is_positive is not True or temperature.is_finite is not True:
        raise ValueError("Temperature must be known positive and finite.")
    logits = sp.Matrix(logits)
    if logits.rows == 0 or logits.cols != 1 or any(
            x.is_real is not True or x.is_finite is not True for x in logits):
        raise ValueError("Use a nonempty column of finite real logits.")
    weights = logits.applyfunc(lambda x: sp.exp(x/temperature))
    return (weights/sum(weights)).applyfunc(sp.simplify)


def restrict(law, indices):
    law = categorical(law)
    ids = tuple(indices)
    if len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 or i >= len(law) for i in ids):
        raise ValueError("Retained indices must be distinct vocabulary indices.")
    mass = sum(law[i] for i in ids)
    if mass <= 0:
        raise ValueError("Retained set must have positive mass.")
    return sp.Matrix([law[i]/mass if i in ids else 0 for i in range(len(law))])


def nucleus(law, rho, tie_order=None):
    law, rho = categorical(law), sp.sympify(rho)
    if rho.is_Rational is not True or not 0 < rho <= 1:
        raise ValueError("Use an exact rational threshold in (0,1].")
    # Full chapter vocabulary uses EOS,A,B,C; the active alphabet uses A,B,C.
    default_order = (3, 0, 1, 2) if len(law) == len(TOKENS) else tuple(range(len(law)))
    order = default_order if tie_order is None else tuple(tie_order)
    if sorted(order) != list(range(len(law))):
        raise ValueError("Tie order must permute the vocabulary indices.")
    rank = {i: j for j, i in enumerate(order)}
    sorted_ids = sorted(range(len(law)), key=lambda i: (-law[i], rank[i]))
    retained, mass = [], sp.Integer(0)
    for i in sorted_ids:
        retained.append(i)
        mass += law[i]
        if mass >= rho:
            break
    return restrict(law, retained)


@dataclass(frozen=True)
class Verification:
    acceptance: sp.Matrix
    mass: sp.Matrix
    overlap: sp.Expr
    residual: sp.Matrix | None
    output: sp.Matrix
    coupling: sp.Matrix


def verify(target, proposal):
    target, proposal = categorical(target), categorical(proposal)
    if target.shape != proposal.shape:
        raise ValueError("Target and proposal vocabularies must agree.")
    mass = sp.Matrix([min(a, b) for a, b in zip(target, proposal)])
    alpha = sum(mass)
    acceptance = sp.Matrix([min(1, a/b) if b > 0 else 0 for a, b in zip(target, proposal)])
    residual = None if alpha == 1 else (target-mass)/(1-alpha)
    output = mass if residual is None else mass+(1-alpha)*residual
    coupling = sp.diag(*mass)
    if alpha < 1:
        coupling += (proposal-mass)*(target-mass).T/(1-alpha)
    return Verification(acceptance, mass, alpha, residual, output, coupling)


def pair_law(stale=False, *, proposal_rows=None):
    """Enumerate every draft pair and positive accept/reject/output branch.

    First rejection uses a target-only restart at the corrected prefix.
    stale=True changes only that branch to emit its old drafted suffix.
    """
    proposal_rows = Q if proposal_rows is None else proposal_rows
    first = verify(p, q)
    joint, branches = sp.zeros(3), []
    for x, y in product(range(3), repeat=2):
        draft = q[x]*proposal_rows[x, y]
        for b in range(3):
            if first.residual is not None:
                w = draft*(1-first.acceptance[x])*first.residual[b]
                for c in range(3):
                    v = w*(int(c == y) if stale else P[b, c])
                    if v:
                        joint[b, c] += v
                        branches.append((x, y, "reject-1", b, c, v))
        second = verify(P[x, :].T, proposal_rows[x, :].T)
        w = draft*first.acceptance[x]
        accepted = w*second.acceptance[y]
        if accepted:
            joint[x, y] += accepted
            branches.append((x, y, "accept-both", x, y, accepted))
        if second.residual is not None:
            for c in range(3):
                v = w*(1-second.acceptance[y])*second.residual[c]
                if v:
                    joint[x, c] += v
                    branches.append((x, y, "reject-2", x, c, v))
    return joint, branches


def commit(tokens, remaining, eos="EOS"):
    """Emission guard for a currently unterminated prefix; includes first EOS."""
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
        raise ValueError("Remaining budget must be a nonnegative integer.")
    out = []
    for token in tokens:
        if len(out) == remaining:
            break
        out.append(token)
        if token == eos:
            break
    return tuple(out)


def block_law(target, proposal, history, k, remaining, eos):
    """Exact law of one linear-chain iteration; categorical kernels use indices.

    Stops drafting at proposed EOS. A rejected EOS can yield a non-EOS
    correction. No target/proposal kernel is called beyond committed EOS.
    This small enumeration oracle is not a production cache implementation.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("Horizon must be a positive integer.")
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
        raise ValueError("Remaining budget must be a nonnegative integer.")
    history = tuple(history)
    if remaining == 0 or eos in history:
        return {(): sp.Integer(1)}
    horizon = min(k, remaining)
    drafts = []

    def draft(prefix, vectors, weight):
        if len(prefix) == horizon or (prefix and prefix[-1] == eos):
            drafts.append((prefix, vectors, weight))
            return
        law = categorical(proposal(history+prefix))
        for x, mass in enumerate(law):
            if mass:
                draft(prefix+(x,), vectors+(law,), weight*mass)

    draft((), (), sp.Integer(1))
    out = defaultdict(lambda: sp.Integer(0))
    for candidates, vectors, weight in drafts:
        active = weight
        for i, x in enumerate(candidates):
            h = history+candidates[:i]
            verification = verify(target(h), vectors[i])
            if verification.residual is not None:
                for y, mass in enumerate(verification.residual):
                    if mass:
                        out[candidates[:i]+(y,)] += active*(1-verification.acceptance[x])*mass
            active *= verification.acceptance[x]
            if active == 0:
                break
        if active:
            if candidates[-1] == eos or len(candidates) == remaining:
                out[candidates] += active
            else:
                law = categorical(target(history+candidates))
                for y, mass in enumerate(law):
                    if mass:
                        out[candidates+(y,)] += active*mass
    return {seq: mass for seq, mass in out.items() if mass}


def complete_law(target, proposal, k, budget, eos, history=()):
    """Repeat correct iterations until EOS or budget; finite exhaustive oracle."""
    history = tuple(history)
    if budget == 0 or eos in history:
        return {history: sp.Integer(1)}
    out = defaultdict(lambda: sp.Integer(0))
    for block, mass in block_law(target, proposal, history, k, budget, eos).items():
        for seq, rest in complete_law(target, proposal, k, budget-len(block), eos, history+block).items():
            out[seq] += mass*rest
    return dict(out)


def target_law(target, budget, eos, history=()):
    history = tuple(history)
    if budget == 0 or eos in history:
        return {history: sp.Integer(1)}
    out = {}
    for x, mass in enumerate(categorical(target(history))):
        if mass:
            for seq, rest in target_law(target, budget-1, eos, history+(x,)).items():
                out[seq] = mass*rest
    return out


def cost_grid():
    alpha = R(5, 6)
    return [(k, sum(alpha**j for j in range(k+1)), R(11, 10)+R(3, 20)*k)
            for k in range(1, 9)]


def main():
    checks = []

    def equal(actual, expected, label):
        if isinstance(actual, sp.MatrixBase) or isinstance(expected, sp.MatrixBase):
            a, b = sp.Matrix(actual), sp.Matrix(expected)
            ok = a.shape == b.shape and all(sp.simplify(x) == 0 for x in a-b)
        elif isinstance(actual, dict):
            ok = actual.keys() == expected.keys() and all(sp.simplify(actual[x]-expected[x]) == 0 for x in actual)
        else:
            ok = sp.simplify(actual-expected) == 0
        if not ok:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        checks.append(label)

    def require(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    # A: exact readouts, transformation order, fixed-support entropy.
    equal(WT*o_star, sp.Matrix([sp.log(3), sp.log(2), 0, 0]), "A target readout")
    equal(WD*o_star, sp.Matrix([sp.log(2), sp.log(3), 0, 0]), "A proposal readout")
    equal(softmax(WT*o_star), sp.Matrix([R(3, 7), R(2, 7), R(1, 7), R(1, 7)]), "A unmasked target")
    equal(softmax(WD*o_star), sp.Matrix([R(2, 7), R(3, 7), R(1, 7), R(1, 7)]), "A unmasked draft")
    equal(restrict(softmax(WT*o_star), (0, 1, 2)), p.col_join(sp.zeros(1, 1)), "A mask target")
    equal(restrict(softmax(WD*o_star), (0, 1, 2)), q.col_join(sp.zeros(1, 1)), "A mask draft")
    active_logits = sp.Matrix([sp.log(3), sp.log(2), 0])
    half = softmax(active_logits, R(1, 2))
    equal(half, sp.Matrix([R(9, 14), R(2, 7), R(1, 14)]), "A half temperature")
    equal(restrict(p, (0, 1)), sp.Matrix([R(3, 5), R(2, 5), 0]), "A top two")
    equal(nucleus(p, R(3, 4)), restrict(p, (0, 1)), "A nucleus three quarters")
    equal(nucleus(p, R(1, 2)), sp.Matrix([1, 0, 0]), "A threshold equality")
    equal(nucleus(half, R(3, 5)), sp.Matrix([1, 0, 0]), "A temperature then nucleus")
    powered = nucleus(p, R(3, 5)).applyfunc(lambda x: x**2)
    equal(powered/sum(powered), sp.Matrix([R(9, 13), R(4, 13), 0]), "A nucleus then temperature")
    equal(nucleus(softmax(WT*o_star), R(6, 7)), sp.Matrix([R(1, 2), R(1, 3), 0, R(1, 6)]), "A full-vocabulary EOS-first tie order")
    t = sp.symbols("T", positive=True)
    weights = active_logits.applyfunc(lambda x: sp.exp(x/t))
    Z, probs = sum(weights), weights/sum(weights)
    mu = sum(a*b for a, b in zip(probs, active_logits))
    H = sp.log(Z)-mu/t
    variance = sum(a*b*b for a, b in zip(probs, active_logits))-mu**2
    equal(sp.diff(H, t)-variance/t**3, 0, "A entropy derivative residual")
    equal(-sum(a*sp.log(a) for a in probs)-H, 0, "A entropy identity")
    shift = sp.symbols("c", real=True, finite=True)
    equal(softmax(active_logits+sp.ones(3, 1)*shift), p, "A common-logit shift")
    equal(sp.limit(1/(2+sp.exp(-1/t)), t, 0, dir="+"), R(1, 2), "A tied maximum limit")

    # B: all support cases and independent marginal checks.
    cases = [(p, q), (p, p), (p, sp.Matrix([1, 0, 0])),
             (sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]))]
    for i, (target, proposal) in enumerate(cases):
        v = verify(target, proposal)
        equal(v.output, target, f"B case {i} output")
        equal(v.coupling*sp.ones(len(target), 1), proposal, f"B case {i} draft marginal")
        equal(v.coupling.T*sp.ones(len(target), 1), target, f"B case {i} target marginal")
        equal(sp.trace(v.coupling), v.overlap, f"B case {i} maximal agreement")
        equal(v.overlap, 1-sum(abs(a-b) for a, b in zip(target, proposal))/2, f"B case {i} TV")
        if v.residual is not None:
            equal(sum(v.residual), 1, f"B case {i} residual normalization")
    v = verify(p, q)
    equal(v.acceptance, sp.Matrix([1, R(2, 3), 1]), "B per-token acceptance")
    equal(v.overlap, R(5, 6), "B overlap")
    equal(v.residual, sp.Matrix([1, 0, 0]), "B correction")
    require(verify(p, p).residual is None, "B unreachable equal-law residual")
    equal(verify(p, [1, 0, 0]).residual, sp.Matrix([0, R(2, 3), R(1, 3)]), "B missing support supplied")
    # Exhaust all rational categorical pairs with denominator four, on 3 tokens.
    lattice = [sp.Matrix([R(a, 4), R(b, 4), R(4-a-b, 4)])
               for a in range(5) for b in range(5-a)]
    for i, target in enumerate(lattice):
        for j, proposal in enumerate(lattice):
            equal(verify(target, proposal).output, target, f"B rational simplex pair {i}/{j}")

    # C: all nine outcomes, all positive branches, two controlled bugs.
    joint, branches = pair_law()
    expected = sp.diag(*p)*P
    for a, b in product(range(3), repeat=2):
        equal(joint[a, b], expected[a, b], f"C joint entry {a}/{b}")
    equal(sum(branch[-1] for branch in branches), 1, "C exhaustive branch mass")
    require({(b[0], b[1]) for b in branches} == set(product(range(3), repeat=2)), "C all nine draft pairs")
    stale, stale_branches = pair_law(stale=True)
    equal(sum(b[-1] for b in stale_branches), 1, "C stale branch mass")
    equal(stale[0, 1]-expected[0, 1], R(1, 72), "C stale AB excess")
    equal(stale[0, 2]-expected[0, 2], -R(1, 72), "C stale AC deficit")
    naive = v.mass+(1-v.overlap)*p
    equal(naive, sp.Matrix([R(5, 12), R(7, 18), R(7, 36)]), "C naive target replacement law")
    require(naive != p and stale != expected, "C both incorrect algorithms detected")

    # D: reached-position survival and the separate stationary grid.
    overlaps = sp.Matrix([verify(P[i, :].T, Q[i, :].T).overlap for i in range(3)])
    equal(overlaps, sp.Matrix([R(3, 4), R(3, 4), 1]), "D conditional row overlaps")
    s1, s2 = v.overlap, v.mass.dot(overlaps)
    equal(s2, R(2, 3), "D two-proposal survival")
    equal(sp.Matrix([1-s1, s1-s2, s2]), sp.Matrix([R(1, 6), R(1, 6), R(2, 3)]), "D accepted-prefix distribution")
    equal(s2/s1, R(4, 5), "D reached second rate")
    equal(s1+s2, R(3, 2), "D accepted count")
    equal(1+s1+s2, R(5, 2), "D emitted count")
    first_accepted = v.mass / s1
    equal(first_accepted, sp.Matrix([R(2, 5), R(2, 5), R(1, 5)]),
          "A4 law of the first token given acceptance")
    equal(sum(b[-1] for b in branches if b[2] == "accept-both"), s2,
          "A4 original survival from independent branch enumeration")
    changed_rows = Q.copy()
    changed_rows[2, :] = sp.Matrix([[1, 0, 0]])
    changed_joint, changed_branches = pair_law(proposal_rows=changed_rows)
    equal(changed_joint, expected, "A4 changing only proposal preserves emitted pair law")
    changed_survival = sum(b[-1] for b in changed_branches if b[2] == "accept-both")
    changed_overlap = verify(P[2, :].T, changed_rows[2, :].T).overlap
    equal(changed_survival, v.mass.dot(sp.Matrix([overlaps[0], overlaps[1], changed_overlap])),
          "A4 changed survival from independent branch enumeration")
    equal(changed_survival, R(13, 24), "A4 changed both-accept mass")
    equal(changed_survival/s1, R(13, 20), "A4 changed reached second rate")
    equal(s1+changed_survival, R(11, 8), "A4 changed accepted count")
    equal(1+s1+changed_survival, R(19, 8), "A4 changed progress")
    require(s2 != s1**2 and changed_survival != s1**2, "A4 stationary rate is a separate model")
    alpha = sp.symbols("alpha", nonnegative=True)
    j = sp.symbols("j", integer=True, nonnegative=True)
    for k in range(1, 9):
        equal(sum(alpha**i for i in range(k+1))*(1-alpha), 1-alpha**(k+1), f"D geometric identity k={k}")
        equal(sum(sp.Integer(1)**i for i in range(k+1)), k+1, f"D alpha one k={k}")
    grid = cost_grid()
    winner = max(grid, key=lambda row: row[1]/row[2])
    equal(winner[0], 6, "D unique maximizing horizon")
    require(sum(row[1]/row[2] == winner[1]/winner[2] for row in grid) == 1, "D optimum is unique")
    equal(winner[1]/winner[2], R(201811, 93312), "D exact proxy optimum")
    for (k, n, c), (_, n1, c1) in zip(grid, grid[1:]):
        equal(n1-n, R(5, 6)**(k+1), f"D nested increment {k}")
        require(bool(n1/c1 > n/c) == bool((n1-n)*c > n*(c1-c)), f"D marginal criterion {k}")
    equal(grid[1][1]/grid[1][2], R(65, 36), "D cheap stationary drafter k2")
    equal(sp.Integer(3)/(2*R(1, 2)+R(11, 10)+R(1, 10)), R(15, 11), "D slower perfect drafter k2")

    # E: marginals versus joint and differentiated weighted CE.
    binary = sp.Matrix([[R(1, 2), 0], [0, R(1, 2)]])
    independent = sp.ones(2)*R(1, 4)
    equal(binary*sp.ones(2, 1), independent*sp.ones(2, 1), "E first marginals")
    equal(binary.T*sp.ones(2, 1), independent.T*sp.ones(2, 1), "E second marginals")
    require(binary != independent, "E joint mismatch")
    equal(independent[0, 1]+independent[1, 0], R(1, 2), "E forbidden proposal mass")
    for i in range(2):
        target = sp.eye(2)[:, i]
        v2 = verify(target, [R(1, 2), R(1, 2)])
        equal(v2.overlap, R(1, 2), f"E second overlap after {i}")
        equal(v2.output, target, f"E corrected conditional after {i}")
    for head in range(2):
        logits = sp.Matrix(sp.symbols(f"v{head}A v{head}B", real=True))
        lam = sp.Integer(1) if head == 0 else sp.symbols("lambda", nonnegative=True)
        probs = softmax(logits)
        loss = -lam*sp.log(probs[0])
        for x in range(2):
            equal(sp.diff(loss, logits[x]), lam*(probs[x]-int(x == 0)), f"E head {head} gradient {x}")

    # F: selection, validation and complete EOS-aware sequence oracles.
    uniform = sp.Matrix([R(1, 2), R(1, 2)])
    equal(verify(uniform, uniform).acceptance[0], 1, "F stale denominator accepts selected A")
    equal(verify(uniform, [1, 0]).output, uniform, "F selected-law repair")
    bad_inputs = [[-R(1, 2), R(3, 2)], [R(1, 3), R(1, 3)], [0.5, 0.5], [], [[1, 0]]]
    for i, bad in enumerate(bad_inputs):
        try:
            categorical(bad)
        except ValueError:
            checks.append(f"F rejected invalid categorical {i}")
        else:
            raise AssertionError("Invalid categorical accepted")
    try:
        verify([1, 0], [1, 0, 0])
    except ValueError:
        checks.append("F inconsistent vocabularies rejected")
    else:
        raise AssertionError("Inconsistent vocabularies accepted")
    require(commit(["EOS", "A", "B"], 3) == ("EOS",), "F first-token EOS")
    require(commit(["A", "B", "EOS"], 1) == ("A",), "F remaining budget one")
    require(commit(["A", "EOS", "B"], 3) == ("A", "EOS"), "F middle EOS")
    require(commit(["A"], 0) == (), "F zero budget")

    def teaching_kernel(history, draft=False):
        if 3 in history:
            raise AssertionError("Requested conditional after EOS")
        if len(history) == 0:
            law = q if draft else p
        elif len(history) == 1:
            law = (Q if draft else P)[history[0], :].T
        elif len(history) == 2:
            law = q if draft else p
        else:
            return sp.Matrix([0, 0, 0, 1])
        return law.col_join(sp.zeros(1, 1))

    target = lambda h: teaching_kernel(h)
    proposal = lambda h: teaching_kernel(h, True)
    for k, budget in product(range(1, 4), range(1, 5)):
        law = complete_law(target, proposal, k, budget, 3)
        equal(law, target_law(target, budget, 3), f"F full teaching law k={k}, budget={budget}")
        equal(sum(law.values()), 1, f"F teaching mass k={k}, budget={budget}")
    # This target can stop at the first token; the draft can propose EOS and
    # have it rejected. Strict kernels catch all post-termination queries.
    def early_target(h):
        if 1 in h:
            raise AssertionError("Target called after EOS")
        return uniform if not h else sp.Matrix([0, 1])

    def early_proposal(h):
        if 1 in h:
            raise AssertionError("Proposal called after EOS")
        return sp.Matrix([R(1, 4), R(3, 4)]) if not h else uniform

    for k, budget in product(range(1, 4), range(1, 4)):
        equal(complete_law(early_target, early_proposal, k, budget, 1),
              target_law(early_target, budget, 1), f"F early/rejected EOS k={k}, budget={budget}")

    return {"python": sys.version.split()[0], "sympy": sp.__version__,
            "checks_passed": len(checks), "scope": "Exact finite laws and symbolic derivatives; no GPU benchmark or production implementation certification.",
            "finite_domains": {"rational_simplex_vectors": len(lattice), "rational_pairs": len(lattice)**2,
                               "draft_pairs": 9, "positive_correct_branches": len(branches),
                               "horizons": list(range(1, 9)), "full_teaching_horizon_budget_cases": 12,
                               "early_eos_horizon_budget_cases": 9},
            "joint": [[str(x) for x in joint.row(i)] for i in range(3)],
            "stale_joint": [[str(x) for x in stale.row(i)] for i in range(3)],
            "accepted_prefix": ["1/6", "1/6", "2/3"],
            "A4_first_accepted_token_law": list(map(str, first_accepted)),
            "A4_changed_proposal_after_C": {
                "both_accepted": str(changed_survival),
                "reached_second_rate": str(changed_survival/s1),
                "expected_accepted": str(s1+changed_survival),
                "expected_progress": str(1+s1+changed_survival)},
            "cost_grid": [{"k": k, "N": str(n), "C": str(c), "proxy": str(n/c)} for k, n, c in grid],
            "checks": checks}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
