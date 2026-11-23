"""Exact teaching fixtures for Chapter 18; no sampled training or model downloads."""

from itertools import product
import json
import platform

import sympy as sp

I, Q = sp.Integer, sp.Rational


def tree(p, q, c):
    """D5, D4, V5, V4; direct-answer continuation is fixed."""
    probabilities = sp.Matrix([(1 - p) / 2, (1 - p) / 2,
                               p * q, p * (1 - q)])
    rewards = sp.Matrix([1, 0, 1 - c, -c])
    scores = sp.Matrix([[-p, 0], [-p, 0], [1 - p, 1 - q], [1 - p, -q]])
    return probabilities, rewards, scores


def weighted_score(probabilities, coefficients, scores):
    return sp.simplify(scores.T * sp.diag(*probabilities) * coefficients)


def group_coefficients(rewards):
    """Population standard deviation; the teaching zero-variance rule is exact."""
    k = len(rewards)
    if k < 2:
        raise ValueError("A leave-one-out group requires K >= 2")
    mean = sum(rewards) / I(k)
    centered = sp.Matrix([r - mean for r in rewards])
    std = sp.sqrt(sum(a ** 2 for a in centered) / I(k))
    standardized = sp.zeros(k, 1) if std == 0 else centered / std
    loo = centered * Q(k, k - 1)
    return mean, std, centered, standardized, loo


def clipped_surrogate(ratio, advantage, epsilon=Q(1, 5)):
    clipped = sp.Min(sp.Max(ratio, 1 - epsilon), 1 + epsilon)
    return sp.Min(ratio * advantage, clipped * advantage)


def reductions(sequences):
    """Sequence sum average, response mean average, global token mean."""
    if not sequences or any(not seq for seq in sequences):
        raise ValueError("Reductions require nonempty included responses")
    sums = [sum(seq) for seq in sequences]
    return (sum(sums) / I(len(sequences)),
            sum(total / I(len(seq)) for total, seq in zip(sums, sequences))
            / I(len(sequences)),
            sum(sums) / I(sum(map(len, sequences))))


def importance_weights(target, behavior):
    """Reject uncovered target support instead of silently discarding it."""
    if len(target) != len(behavior):
        raise ValueError("Probability vectors must have the same domain")
    weights = []
    for t, b in zip(target, behavior):
        if t < 0 or b < 0:
            raise ValueError("Probabilities must be nonnegative")
        if b == 0 and t != 0:
            raise ValueError("Target support is not covered by behavior")
        weights.append(I(0) if b == 0 else sp.cancel(t / b))
    if sum(target) != 1 or sum(behavior) != 1:
        raise ValueError("Probabilities must sum to one")
    return sp.Matrix(weights)


def td_target(reward, next_value, gamma=I(1), *, terminated=False,
              truncated=False):
    """Truncation ends storage, not the bootstrap; true termination zeroes it."""
    return reward + gamma * (0 if terminated else next_value)


def gae(rewards, values, terminated, gamma=I(1), lam=Q(1, 2)):
    """One non-crossing segment; final values specify its bootstrap convention."""
    if len(values) != len(rewards) + 1 or len(terminated) != len(rewards):
        raise ValueError("Need T rewards/flags and T+1 values")
    if any(terminated[:-1]):
        raise ValueError("Split episodes before computing GAE")
    deltas = [td_target(r, values[t + 1], gamma, terminated=terminated[t])
              - values[t] for t, r in enumerate(rewards)]
    advantages = [sum((gamma * lam) ** (u - t) * deltas[u]
                      for u in range(t, len(rewards)))
                  for t in range(len(rewards))]
    return sp.Matrix(deltas), sp.Matrix(advantages)


def verify():
    """Return a named assertion summary; assert exact identities before numerics."""
    passed = []

    def check(name, actual, expected):
        residual = sp.simplify(actual - expected)
        zero = sp.zeros(*residual.shape) if isinstance(residual, sp.MatrixBase) else 0
        if residual != zero:
            raise AssertionError(f"{name}: {actual} != {expected}")
        passed.append(name)

    s = sp.Matrix([0, sp.log(2), sp.log(3)])
    a = s.applyfunc(sp.exp) / sum(s.applyfunc(sp.exp))
    o_star = sp.Matrix([[1, 0, 1], [0, 1, 1]]) * a
    check("A.attention_normalized", sum(a), I(1))
    check("A.attention_weights", a, sp.Matrix([Q(1, 6), Q(1, 3), Q(1, 2)]))
    check("A.feature", o_star, sp.Matrix([Q(2, 3), Q(5, 6)]))
    check("A.feature_norm", o_star.dot(o_star), Q(41, 36))
    p, q = sp.symbols("p q", positive=True)  # additionally p,q < 1
    c, b = sp.symbols("c b", real=True)
    P, R, S = tree(p, q, c)
    J = sp.expand(P.dot(R))
    direct = sp.Matrix([sp.diff(J, p) * p * (1 - p),
                        sp.diff(J, q) * q * (1 - q)])
    check("A.tree_normalized", sum(P), I(1))
    check("A.objective", J, Q(1, 2) + p * (q - c - Q(1, 2)))
    check("B.score_gradient", weighted_score(P, R, S), direct)
    check("B.zero_mean_score", S.T * P, sp.zeros(2, 1))
    check("B.constant_baseline", weighted_score(P, R - sp.ones(4, 1) * b, S), direct)
    initial = {p: Q(1, 2), q: Q(3, 4), c: Q(1, 10)}
    Pi, Ri, Si = P.subs(initial), R.subs(initial), S.subs(initial)
    g = direct.subs(initial)
    check("A.initial_paths", Pi, sp.Matrix([Q(1, 4), Q(1, 4), Q(3, 8), Q(1, 8)]))
    check("A.correctness", Pi.dot(sp.Matrix([1, 0, 1, 0])), Q(5, 8))
    check("A.tool_count", Pi.dot(sp.Matrix([0, 0, 1, 1])), Q(1, 2))
    check("A.tool_cost", (c * p).subs(initial), Q(1, 20))
    check("A.utility", J.subs(initial), Q(23, 40))
    check("B.logit_gradient", g, sp.Matrix([Q(3, 80), Q(3, 32)]))
    check("B.readout_gradient", g[0] * o_star, sp.Matrix([Q(1, 40), Q(1, 32)]))
    # Causality removes already-paid cost only from the second action.
    G2 = sp.Matrix([1, 0, 1, 0])
    check("B.return_to_go_phi", sum(P[i] * G2[i] * S[i, 1] for i in range(4)), direct[1])
    for j in range(4):
        for coordinate in range(2):
            derivative = sp.diff(P[j], (p, q)[coordinate])
            derivative *= (p, q)[coordinate] * (1 - (p, q)[coordinate])
            check(f"B.leaf_score_{j}_{coordinate}", derivative, P[j] * S[j, coordinate])
    mean_h = p * (1 - p)
    second = p * ((1 - b) * (1 - p)) ** 2 + (1 - p) * (b * p) ** 2
    check("B.baseline_variance", second - mean_h ** 2, p * (1 - p) * (1 - p - b) ** 2)
    check("B.optimal_baseline", sp.diff(second, b).subs(b, 1 - p), I(0))
    check("B.action_dependent_baseline", weighted_score(P, R - R, S), sp.zeros(2, 1))
    # Outcome-only fixture is a separate c=0 experiment.
    group = [0, 2, 1, 2]
    outcome = sp.Matrix([1, 0, 1, 0])
    rewards = [outcome[i] for i in group]
    mean, std, center, standard, loo = group_coefficients(rewards)
    check("C.mean", mean, Q(3, 4))
    check("C.std", std, sp.sqrt(3) / 4)
    check("C.center", center, sp.Matrix([Q(1, 4), Q(1, 4), -Q(3, 4), Q(1, 4)]))
    check("C.loo", loo, sp.Matrix([Q(1, 3), Q(1, 3), -1, Q(1, 3)]))
    check("C.standard", standard, sp.Matrix([1, 1, -3, 1]) / sp.sqrt(3))
    sample_scores = Si[group, :]
    check("C.center_gradient", sample_scores.T * center / 4, sp.Matrix([Q(1, 8), Q(1, 32)]))
    check("C.loo_gradient", sample_scores.T * loo / 4, sp.Matrix([Q(1, 6), Q(1, 24)]))
    for value in [I(0), I(1), Q(2, 3)]:
        stats = group_coefficients([value] * 4)
        for choice in (2, 3, 4):
            check(f"C.equal_{value}_{choice}", stats[choice], sp.zeros(4, 1))
    mass, expected_center, expected_loo = I(0), sp.zeros(2, 1), sp.zeros(2, 1)
    expected_standard = sp.zeros(2, 1)
    count = 0
    for indices in product(range(4), repeat=4):
        probability = sp.prod(Pi[i] for i in indices)
        stats = group_coefficients([outcome[i] for i in indices])
        scores = Si[list(indices), :]
        mass += probability
        expected_center += probability * scores.T * stats[2] / 4
        expected_loo += probability * scores.T * stats[4] / 4
        expected_standard += probability * scores.T * stats[3] / 4
        count += 1
    true_outcome = direct.subs(initial).subs(c, 0)  # compute separately below
    true_outcome = direct.subs(c, 0).subs(initial)
    check("C.enumeration_count", I(count), I(256))
    check("C.enumeration_mass", mass, I(1))
    check("C.self_inclusion_expectation", expected_center, Q(3, 4) * true_outcome)
    check("C.loo_expectation", expected_loo, true_outcome)
    if sp.simplify(expected_standard - true_outcome) == sp.zeros(2, 1):
        raise AssertionError("Standardization unexpectedly equals the true gradient")
    passed.append("C.standardization_changes_expectation")
    rho = sp.symbols("rho", positive=True)
    for k, (ratio, advantage, result, slope) in enumerate([
            (Q(3, 2), Q(1, 4), Q(3, 10), 0),
            (Q(3, 2), -Q(3, 4), -Q(9, 8), -Q(3, 4)),
            (Q(1, 2), Q(1, 4), Q(1, 8), Q(1, 4)),
            (Q(1, 2), -Q(3, 4), -Q(3, 5), 0)]):
        check(f"D.clip_{k}", clipped_surrogate(ratio, advantage), result)
        branch = sp.Min(rho, Q(6, 5)) * advantage if advantage > 0 else sp.Max(rho, Q(4, 5)) * advantage
        check(f"D.clip_slope_{k}", sp.diff(branch, rho).subs(rho, ratio), I(slope) if slope == 0 else slope)
    current, ref, old = [Q(3, 4), Q(1, 4)], [Q(1, 4), Q(3, 4)], [Q(1, 2)] * 2
    kl = sum(t * sp.log(t / r) for t, r in zip(current, ref))
    penalties = [r / t - sp.log(r / t) - 1 for t, r in zip(current, ref)]
    check("D.binary_KL", kl, sp.log(3) / 2)
    check("D.current_sampled_KL", sum(t * k for t, k in zip(current, penalties)), kl)
    check("D.old_sampled_penalty", sum(t * k for t, k in zip(old, penalties)), Q(2, 3))
    for i, (actual, expected) in enumerate(zip(reductions([[I(1)] * 2, [-I(1)] * 4]), [-1, 0, -Q(1, 3)])):
        check(f"D.reduction_{i}", actual, expected)
    owners = (1, 0, 1)
    factors = (p, q, q)  # the middle q is the deliberately wrong observation score
    check("E.owned_likelihood", sp.prod(f ** m for f, m in zip(factors, owners)), p * q)
    check("E.wrong_extra_factor", sp.prod(factors) / (p * q), q)
    behavior = P.subs({p: Q(1, 2), q: Q(1, 2)})
    weights = importance_weights(Pi, behavior)
    check("E.weights", weights, sp.Matrix([1, 1, Q(3, 2), Q(1, 2)]))
    check("E.importance_utility", sum(behavior[i] * weights[i] * Ri[i] for i in range(4)), Q(23, 40))
    check("E.importance_gradient", weighted_score(behavior, sp.diag(*weights) * Ri, Si), g)
    check("E.ESS", sum(weights) ** 2 / sum(w ** 2 for w in weights), Q(32, 9))
    try:
        importance_weights([Q(1, 2), Q(1, 2)], [I(1), I(0)])
    except ValueError:
        passed.append("E.missing_support_rejected")
    else:
        raise AssertionError("Missing support was accepted")
    check("E.shared_zero_support", importance_weights([I(1), I(0)], [I(1), I(0)]), sp.Matrix([1, 0]))
    rewards, values = [I(0), I(0), I(1)], [Q(1, 4), Q(1, 2), Q(3, 4), I(0)]
    deltas, advantages = gae(rewards, values, [False, False, True])
    check("F.returns", sp.Matrix([sum(rewards[t:]) for t in range(3)]), sp.ones(3, 1))
    check("F.deltas", deltas, sp.ones(3, 1) / 4)
    check("F.GAE_half", advantages, sp.Matrix([Q(7, 16), Q(3, 8), Q(1, 4)]))
    check("F.GAE_one", gae(rewards, values, [False, False, True], lam=I(1))[1],
          sp.Matrix([Q(3, 4), Q(1, 2), Q(1, 4)]))
    check("F.cutoff_target", td_target(I(0), Q(3, 4), truncated=True), Q(3, 4))
    check("F.terminal_target", td_target(I(0), Q(3, 4), terminated=True), I(0))
    gamma = sp.symbols("gamma", nonnegative=True)
    potential = sp.symbols("Phi1:5", real=True)
    shaped = sum(gamma ** t * (gamma * potential[t + 1] - potential[t]) for t in range(3))
    check("F.shaping", shaped, -potential[0] + gamma ** 3 * potential[3])
    l1, l2, l3 = sp.symbols("l1 l2 l3", real=True)
    check("F.fragment_additivity", (l1 + l2) + l3, l1 + l2 + l3)
    check("F.fragment_mean_counterexample", ((I(1) + 1) / 2 - I(1)) / 2, I(0))
    check("F.whole_fragment_mean", (I(1) + 1 - 1) / 3, Q(1, 3))
    w_plus = Q(1, 10) * g[0] * o_star
    alpha_plus = w_plus.dot(o_star)
    phi_plus = sp.log(3) + Q(1, 10) * g[1]
    check("B.w_plus", w_plus, sp.Matrix([Q(1, 400), Q(1, 320)]))
    check("B.alpha_plus", alpha_plus, Q(41, 9600))
    check("B.phi_plus", phi_plus, sp.log(3) + Q(3, 320))
    p_plus, q_plus = 1 / (1 + sp.exp(-alpha_plus)), 1 / (1 + sp.exp(-phi_plus))
    j_plus = J.subs({p: p_plus, q: q_plus, c: Q(1, 10)})
    if abs(j_plus.evalf(40) - sp.Float("0.5760388731555395", 40)) > sp.Float("1e-16", 40):
        raise AssertionError("Numerical utility checkpoint failed")
    if not j_plus.evalf(40) > Q(23, 40):
        raise AssertionError("Declared finite-tree update did not improve utility")
    return {"python": platform.python_version(), "sympy": sp.__version__,
            "exact_checks": len(passed), "named_checks": passed, "ordered_groups": count,
            "numerical_checks": 1,
            "p_plus": str(p_plus.evalf(30)), "q_plus": str(q_plus.evalf(30)),
            "J_plus": str(j_plus.evalf(30)),
            "standardized_expectation": str(sp.simplify(expected_standard))}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
