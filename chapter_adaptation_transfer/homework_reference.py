"""Exact reference for merged Chapter 17's four-part SymPy assignment.

Constructed inputs only. The general factor, chain-rule and KL-gap arguments
remain written proofs in the chapter, rather than consequences of finite tests.
"""
import json
import platform

import sympy as sp

import preferences as pref

R = sp.Rational


def kl_interior(p, q):
    """Finite KL for aligned positive laws; symbolic domains are caller-owned."""
    if len(p) != len(q):
        raise ValueError("KL requires aligned event counts.")
    return sum(pi * sp.log(pi / qi) for pi, qi in zip(p, q))


def run_homework():
    checks = pref.Checks()
    report = {}

    # A. Differentiate the actual soft-target loss, not a memorized update.
    feature = sp.Matrix([R(2, 3), R(5, 6)])
    teacher = sp.Matrix([R(1, 2), R(1, 3), R(1, 6)])
    base = sp.Matrix([[1, 0], [1, 0], [1, 0]])
    A = sp.Matrix(1, 2, sp.symbols('a0:2', real=True))
    B = sp.Matrix(sp.symbols('b0:3', real=True))
    z = (base + B * A) * feature
    loss = sp.log(sum(sp.exp(zi) for zi in z)) - teacher.dot(z)
    dA = sp.Matrix(1, 2, [sp.diff(loss, ai) for ai in A])
    dB = sp.Matrix([sp.diff(loss, bi) for bi in B])
    G = (pref.softmax(z) - teacher) * feature.T
    checks.equal('A general factor residual', dB - G * A.T, sp.zeros(3, 1))
    initial = dict(zip(list(A) + list(B), [1, 0, 0, 0, 0]))
    dA0, dB0 = dA.subs(initial), dB.subs(initial)
    checks.equal('A initial A gradient', dA0, sp.zeros(1, 2))
    checks.equal('A initial B gradient', dB0, sp.Matrix([-R(1, 9), 0, R(1, 9)]))
    A1, B1 = A.subs(initial) - dA0, B.subs(initial) - dB0
    logits1 = (base + B1 * A1) * feature
    checks.equal('A simultaneous logits', logits1, sp.Matrix([R(20, 27), R(18, 27), R(16, 27)]))
    zero = dict.fromkeys(list(A) + list(B), 0)
    checks.equal('A both-zero A', dA.subs(zero), sp.zeros(1, 2))
    checks.equal('A both-zero B', dB.subs(zero), sp.zeros(3, 1))
    report['A'] = {'A_gradient': dA0.tolist(), 'B_gradient': list(dB0),
                   'first_step_logits': list(logits1), 'gradient_shapes': [list(dA.shape), list(dB.shape)],
                   'interpretation': 'The old factors determine the first step; zero B blocks only A.'}

    # B. Enumerate the law, then compare it with the conditional grouping.
    u = sp.symbols('u', positive=True)  # Also u < 1.
    root_p = sp.Matrix([R(3, 4), R(1, 4)])
    half = sp.ones(2, 1) / 2
    root_q = sp.Matrix([u, 1-u])
    P = sp.Matrix([root_p[0] * half[0], root_p[0] * half[1],
                   root_p[1] * root_p[0], root_p[1] * root_p[1]])
    Q = sp.Matrix([u/2, u/2, (1-u)/2, (1-u)/2])
    Q0 = Q.subs(u, R(1, 2))
    checks.equal('B P normalization', sum(P), 1)
    checks.equal('B Q normalization', sum(Q), 1)
    c, f = kl_interior(half, root_p), kl_interior(root_p, half)
    checks.equal('B forward grouping', kl_interior(P, Q0), f + R(1, 4)*f)
    checks.equal('B reverse grouping', kl_interior(Q0, P), c + R(1, 2)*c)
    root_cost = kl_interior(root_q, root_p)
    live = root_cost + (1-u)*c
    frozen = root_cost + c/2
    checks.equal('B collection values', live.subs(u, R(1, 2)), frozen.subs(u, R(1, 2)))
    checks.equal('B derivative gap', sp.diff(live-frozen, u), -sp.log(R(4, 3))/2)
    checks.equal('B live derivative', sp.diff(live, u).subs(u, R(1, 2)), -sp.log(3)-c)
    checks.equal('B frozen derivative', sp.diff(frozen, u).subs(u, R(1, 2)), -sp.log(3))
    report['B'] = {'teacher_leaves': list(P), 'student_leaves': list(Q0),
                   'derivative_gap': pref.exact(sp.diff(live-frozen, u)),
                   'conditional_costs_and_weights': [
                       {'history': 'root', 'forward_cost': pref.exact(f), 'teacher_weight': 1,
                        'reverse_cost': pref.exact(c), 'student_weight': 1},
                       {'history': 'a', 'forward_cost': 0, 'teacher_weight': R(3, 4),
                        'reverse_cost': 0, 'student_weight': R(1, 2)},
                       {'history': 'b', 'forward_cost': pref.exact(f), 'teacher_weight': R(1, 4),
                        'reverse_cost': pref.exact(c), 'student_weight': R(1, 2)}],
                   'interpretation': 'Increasing u avoids the costly b branch; fixed visits omit that derivative.'}

    # C. Build normalized response scores and compare two learning cases.
    reference, rewards = pref.PI_REF, pref.REWARDS
    normalizer, optimum = pref.tilted_policy(reference, rewards, 1)
    checks.equal('C Z', normalizer, R(17, 12))
    checks.equal('C normalization', sum(optimum), 1)
    checks.equal('C optimum', optimum, sp.Matrix([R(12, 17), R(4, 17), R(1, 17)]))
    objective = optimum.dot(rewards) - pref.kl(optimum, reference)
    checks.equal('C optimal objective', objective, sp.log(R(17, 12)))
    implicit = sp.Matrix([sp.log(pi/qi) for pi, qi in zip(optimum, reference)])
    checks.equal('C common reward offset', implicit-rewards,
                 -sp.ones(3, 1)*sp.log(normalizer))
    for label, policy, expected_margin, expected_loss in [
        ('reference', reference, 0, sp.log(2)),
        ('optimum', optimum, sp.log(2), sp.log(R(3, 2))),
    ]:
        m = pref.margin(policy, reference, 0, 1)
        checks.equal('C margin '+label, m, expected_margin)
        checks.equal('C loss '+label, pref.softplus(-m), expected_loss)
    m = sp.symbols('m', real=True)
    checks.equal('C individual derivative', sp.diff(pref.softplus(-m), m).subs(m, sp.log(2)), -R(1, 3))
    report['C'] = {'unnormalized_weights': list(normalizer*optimum),
                   'Z': normalizer, 'optimal_policy': list(optimum),
                   'implicit_scores': list(implicit), 'optimal_objective': pref.exact(objective),
                   'interpretation': 'Mixed labels balance at the optimum; a single winning observation still favors a larger margin.'}

    # D. Check entire response trees, then score the observed leaves.
    current = pref.complete_response_distribution(pref.CURRENT_TREE)
    fixed = pref.complete_response_distribution(pref.REFERENCE_TREE)
    checks.true('D common support', set(current) == set(fixed))
    checks.equal('D six leaves', len(current), 6)
    checks.equal('D current normalization', sum(current.values()), 1)
    checks.equal('D reference normalization', sum(fixed.values()), 1)
    margins = {}
    for mode, target in [('full', sp.log(4)), ('mean', R(5, 6)*sp.log(2)),
                         ('omit_eos', sp.log(R(4, 3)))]:
        ratios = [pref.sequence_score(p, q, mode=mode)
                  for p, q in zip(pref.CURRENT_TOKENS, pref.REFERENCE_TOKENS)]
        margins[mode] = pref.exact(ratios[0]-ratios[1])
        checks.equal('D margin '+mode, margins[mode], target)
    checks.equal('D full loss', pref.softplus(-margins['full']), sp.log(R(5, 4)))
    checks.equal('D omitted eos loss', pref.softplus(-margins['omit_eos']), sp.log(R(7, 4)))
    policies = []
    for t in [R(1, 3), R(2, 3)]:
        policy = sp.Matrix([3*t/4, t/4, 1-t])
        policies.append(policy)
        checks.equal(f'D family normalization {t}', sum(policy), 1)
        checks.equal(f'D family fitted probability {t}',
                     pref.sigmoid(pref.margin(policy, reference, 0, 1)), R(2, 3))
    checks.true('D different unseen mass', policies[0][2] != policies[1][2])
    report['D'] = {'margins': margins, 'policies': [list(pi) for pi in policies],
                   'unobserved_mass': [pi[2] for pi in policies],
                   'pair_probabilities': [pref.exact(pref.sigmoid(pref.margin(pi, reference, 0, 1)))
                                          for pi in policies],
                   'interpretation': 'A fitted pair score neither specifies unseen mass nor equates modified and complete likelihoods.'}
    report['checks'] = len(checks.names)
    return report


if __name__ == '__main__':
    print('Python:', platform.python_version())
    print('SymPy:', sp.__version__)
    print(json.dumps(run_homework(), indent=2, default=str))
