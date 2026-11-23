"""Preference fixtures retained as supporting calculations for merged Chapter 17.

The current four-part assignment is implemented in homework_reference.py.

This is a constructed categorical/finite-tree laboratory, not an LLM experiment.
Run from the repository root as documented in README.md.
"""
import json
import sys

import sympy as sp

R = sp.Rational
O_STAR = sp.Matrix([R(2, 3), R(5, 6)])
W_REF = sp.Matrix([[R(3, 2) * sp.log(3), 0],
                   [R(3, 2) * sp.log(2), 0], [0, 0]])
PI_REF = sp.Matrix([R(1, 2), R(1, 3), R(1, 6)])
REWARDS = sp.Matrix([sp.log(2), 0, -sp.log(2)])
COUNTS = ((0, 1, 2, 1), (1, 2, 2, 1), (0, 2, 4, 1))
SEQUENCE_HORIZON = 4  # Includes the final eos event.
DISPLAYED_RESPONSES = (('a', 'eos'), ('b', 'c', 'eos'))
# At the four displayed nonterminal nodes, d receives all remaining mass.
# Every branch ending in d then terminates deterministically.
CURRENT_TREE = {
    (): {'a': R(1, 2), 'b': R(1, 4), 'd': R(1, 4)},
    ('a',): {'eos': R(1, 2), 'd': R(1, 2)},
    ('b',): {'c': R(1, 2), 'd': R(1, 2)},
    ('b', 'c'): {'eos': R(1, 2), 'd': R(1, 2)},
    ('d',): {'eos': sp.Integer(1)},
    ('a', 'd'): {'eos': sp.Integer(1)},
    ('b', 'd'): {'eos': sp.Integer(1)},
    ('b', 'c', 'd'): {'eos': sp.Integer(1)},
}
REFERENCE_TREE = {
    (): {'a': R(1, 2), 'b': R(1, 4), 'd': R(1, 4)},
    ('a',): {'eos': R(1, 4), 'd': R(3, 4)},
    ('b',): {'c': R(2, 3), 'd': R(1, 3)},
    ('b', 'c'): {'eos': R(3, 4), 'd': R(1, 4)},
    ('d',): {'eos': sp.Integer(1)},
    ('a', 'd'): {'eos': sp.Integer(1)},
    ('b', 'd'): {'eos': sp.Integer(1)},
    ('b', 'c', 'd'): {'eos': sp.Integer(1)},
}
CURRENT_TOKENS = tuple(tuple(CURRENT_TREE[y[:t]][token]
                             for t, token in enumerate(y))
                       for y in DISPLAYED_RESPONSES)
REFERENCE_TOKENS = tuple(tuple(REFERENCE_TREE[y[:t]][token]
                               for t, token in enumerate(y))
                         for y in DISPLAYED_RESPONSES)


def exact(expression):
    """Simplify real algebra/log identities without forcing unsafe expansions."""
    return sp.simplify(sp.expand_log(sp.sympify(expression), force=False))


def softplus(value):
    return sp.log(1 + sp.exp(value))


def sigmoid(value):
    return 1 / (1 + sp.exp(-value))


def softmax(logits):
    logits = sp.Matrix(logits)
    weights = logits.applyfunc(sp.exp)
    return weights.applyfunc(lambda value: exact(value / sum(weights)))


def distribution(values, *, positive=False):
    values = sp.Matrix(values)
    if values.cols != 1 or values.rows == 0:
        raise ValueError("A distribution must be a nonempty column vector.")
    predicate = 'is_positive' if positive else 'is_nonnegative'
    if any(getattr(value, predicate) is not True for value in values):
        raise ValueError("Declare probability positivity/nonnegativity explicitly.")
    if exact(sum(values) - 1) != 0:
        raise ValueError("Probabilities must sum to one.")
    return values


def tilted_policy(reference, rewards, coefficient):
    """Finite, finite-reward tilt; this helper requires positive reference mass.

    The support-restricted theorem is handled separately, rather than clipping
    zero reference probabilities to small positive numbers.
    """
    reference = distribution(reference, positive=True)
    rewards = sp.Matrix(rewards)
    coefficient = sp.sympify(coefficient)
    if coefficient.is_positive is not True:
        raise ValueError("Declare a strictly positive regularization coefficient.")
    if reference.shape != rewards.shape:
        raise ValueError("Reference and reward vectors must have equal shapes.")
    if any(value.is_real is not True or value.is_finite is not True for value in rewards):
        raise ValueError("Rewards must be declared finite and real.")
    weights = sp.Matrix([reference[i] * sp.exp(rewards[i] / coefficient)
                         for i in range(len(reference))])
    normalizer = exact(sum(weights))
    return normalizer, weights.applyfunc(lambda value: exact(value / normalizer))


def kl(candidate, reference):
    """Directed finite KL with exact boundary and support conventions."""
    candidate, reference = distribution(candidate), distribution(reference)
    if candidate.shape != reference.shape:
        raise ValueError("KL distributions must have equal shapes.")
    terms = []
    for pi, qi in zip(candidate, reference):
        if pi == 0:
            continue
        if qi == 0:
            return sp.oo
        if pi.is_positive is not True or qi.is_positive is not True:
            raise ValueError("Positive mass needs provably positive denominators.")
        terms.append(pi * sp.log(pi / qi))
    return exact(sum(terms))


def reward_likelihood(scores, counts=COUNTS):
    total = sum(wins + losses for _, _, wins, losses in counts)
    return sum(wins * softplus(-(scores[i] - scores[j]))
               + losses * softplus(scores[i] - scores[j])
               for i, j, wins, losses in counts) / total


def margin(candidate, reference, winner, loser, coefficient=1):
    # For symbolic simplex probabilities, the caller owns the stated domain.
    return coefficient * (sp.log(candidate[winner] / reference[winner])
                          - sp.log(candidate[loser] / reference[loser]))


def sequence_score(current, reference, *, mode='full'):
    if len(current) != len(reference) or not current:
        raise ValueError("Aligned nonempty token factors are required.")
    factors = list(zip(current, reference))
    if any(sp.sympify(p).is_positive is not True
           or sp.sympify(q).is_positive is not True for p, q in factors):
        raise ValueError("Token factors must be positive.")
    if mode == 'omit_eos':
        factors = factors[:-1]
    elif mode not in ('full', 'mean'):
        raise ValueError("Unknown scoring convention.")
    score = sum(sp.log(p / q) for p, q in factors)
    return exact(score / len(current) if mode == 'mean' else score)


def complete_response_distribution(tree, horizon=SEQUENCE_HORIZON):
    """Enumerate every eos-terminated leaf of a normalized finite token tree."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("The terminal horizon must be a positive integer.")
    leaves, visited = {}, set()

    def visit(prefix, mass):
        if len(prefix) >= horizon:
            raise ValueError("A response failed to terminate within the horizon.")
        if prefix not in tree:
            raise ValueError(f"Missing conditional distribution at {prefix}.")
        visited.add(prefix)
        conditional = tree[prefix]
        distribution(list(conditional.values()), positive=True)
        for token, probability in conditional.items():
            if not isinstance(token, str) or not token:
                raise ValueError("Tokens must be nonempty strings.")
            response = prefix + (token,)
            next_mass = exact(mass * probability)
            if token == 'eos':
                leaves[response] = next_mass
            else:
                visit(response, next_mass)

    visit((), sp.Integer(1))
    if visited != set(tree):
        raise ValueError("The tree contains an unreachable conditional node.")
    return leaves


class Checks:
    def __init__(self):
        self.names = []

    def equal(self, name, actual, expected=0):
        if isinstance(actual, sp.MatrixBase) or isinstance(expected, sp.MatrixBase):
            delta = sp.Matrix(actual) - sp.Matrix(expected)
            for i, value in enumerate(delta):
                self.equal(f'{name}[{i}]', value)
            return
        result = exact(actual - expected)
        if result != 0:
            raise AssertionError(f'{name}: expected exact zero, got {result}')
        self.names.append(name)

    def true(self, name, condition):
        if condition is not True and condition is not sp.true:
            raise AssertionError(f'{name}: {condition}')
        self.names.append(name)


def run_checks():
    c = Checks()
    report = {}
    # H-A: reference, labels, anchored fit, score Hessian and its shift nullspace.
    c.equal('reference logits', W_REF * O_STAR, sp.Matrix([sp.log(3), sp.log(2), 0]))
    c.equal('reference distribution', softmax(W_REF * O_STAR), PI_REF)
    c.equal('observation count', sum(w + l for _, _, w, l in COUNTS), 11)
    pair_probs = [R(2, 3), R(2, 3), R(4, 5)]
    for (i, j, _, _), prob in zip(COUNTS, pair_probs):
        c.equal(f'preference {i}{j}', sigmoid(REWARDS[i] - REWARDS[j]), prob)
    a, b, shift = sp.symbols('a b shift', real=True)
    scores = sp.Matrix([a, 0, -b])
    likelihood = reward_likelihood(scores)
    fitted = {a: sp.log(2), b: sp.log(2)}
    c.equal('reward shift', reward_likelihood(scores + sp.ones(3, 1) * shift), likelihood)
    for variable in (a, b):
        c.equal(f'anchored derivative {variable}', sp.diff(likelihood, variable).subs(fitted))
    anchored_hessian = sp.hessian(likelihood, (a, b)).subs(fitted).applyfunc(exact)
    c.equal('anchored Hessian', anchored_hessian,
            sp.Matrix([[R(22, 165), R(4, 55)], [R(4, 55), R(22, 165)]]))
    c.true('anchored positive minor', anchored_hessian[0, 0] > 0)
    c.true('anchored positive determinant', anchored_hessian.det() > 0)
    s = sp.Matrix(sp.symbols('s0:3', real=True))
    score_hessian = sp.hessian(reward_likelihood(s), s)
    c.equal('shift null direction', score_hessian * sp.ones(3, 1), sp.zeros(3, 1))
    d = sp.symbols('d', real=True)
    c.equal('loss derivative', sp.diff(softplus(-d), d), -sigmoid(-d))
    c.equal('loss curvature', sp.diff(softplus(-d), d, 2), sigmoid(d) * sigmoid(-d))
    report['A_reference'] = list(PI_REF)
    report['A_reference_logits'] = list(W_REF * O_STAR)
    report['A_pairwise_probabilities'] = pair_probs
    report['A_reward_likelihood'] = likelihood
    report['A_fitted_gradient'] = [exact(sp.diff(likelihood, variable).subs(fitted))
                                   for variable in (a, b)]
    report['A_anchored_Hessian'] = anchored_hessian.tolist()

    # H-B: complete simplex objective, stationarity and reduced curvature.
    normalizer, optimum = tilted_policy(PI_REF, REWARDS, 1)
    c.equal('normalizer', normalizer, R(17, 12))
    c.equal('optimum', optimum, sp.Matrix([R(12, 17), R(4, 17), R(1, 17)]))
    expected_reward = exact(optimum.dot(REWARDS))
    optimal_kl = kl(optimum, PI_REF)
    c.equal('reference reward', PI_REF.dot(REWARDS), sp.log(2) / 3)
    c.equal('optimized reward', expected_reward, R(11, 17) * sp.log(2))
    c.equal('optimized KL', optimal_kl, R(11, 17) * sp.log(2) - sp.log(R(17, 12)))
    c.equal('optimal objective', expected_reward - optimal_kl, sp.log(normalizer))
    u, v = sp.symbols('u v', positive=True)
    simplex = sp.Matrix([u, v, 1-u-v])  # Domain: u+v<1.
    objective = simplex.dot(REWARDS) - sum(simplex[i] * sp.log(simplex[i]/PI_REF[i]) for i in range(3))
    subs = {u: optimum[0], v: optimum[1]}
    for variable in (u, v):
        c.equal(f'simplex stationarity {variable}', sp.diff(objective, variable).subs(subs))
    reduced_hessian = sp.hessian(objective, (u, v)).subs(subs)
    c.true('negative Hessian first minor', reduced_hessian[0, 0] < 0)
    c.true('negative Hessian determinant', reduced_hessian.det() > 0)
    for i, candidate in enumerate([PI_REF, sp.Matrix([R(1,3)]*3), sp.Matrix([R(1,4),R(1,4),R(1,2)])]):
        j = candidate.dot(REWARDS) - kl(candidate, PI_REF)
        c.equal(f'KL gap witness {i}', j, sp.log(normalizer) - kl(candidate, optimum))
    c.equal('boundary KL gap', REWARDS[0] - kl([1,0,0], PI_REF), sp.log(normalizer) - kl([1,0,0], optimum))
    report['B_normalizer'] = normalizer
    report['B_optimal_policy'] = list(optimum)
    report['B_expected_reward_units'] = expected_reward
    report['B_optimal_KL_nats'] = optimal_kl
    report['B_objective_reward_units'] = exact(expected_reward-optimal_kl)
    report['B_reduced_Hessian'] = reduced_hessian.tolist()
    report['B_stationary_conditions'] = [sp.diff(objective, variable)
                                        for variable in (u, v)]

    # H-C: inverse scores, matched preferences and independent aggregate gradients.
    implicit = sp.Matrix([sp.log(optimum[i] / PI_REF[i]) for i in range(3)])
    c.equal('implicit reward shift', implicit, REWARDS-sp.ones(3,1)*sp.log(normalizer))
    c.equal('implicit normalization', sum(PI_REF[i]*sp.exp(implicit[i]) for i in range(3)), 1)
    for (i, j, _, _), probability in zip(COUNTS, pair_probs):
        c.equal(f'DPO probability {i}{j}', sigmoid(margin(optimum,PI_REF,i,j)), probability)
    single = softplus(-margin(optimum, PI_REF, 0, 1))
    c.equal('individual loss at reference', softplus(-margin(PI_REF,PI_REF,0,1)), sp.log(2))
    c.equal('individual loss at optimum', single, sp.log(R(3,2)))
    # Policy logits h_i=log q_i + score_i yield the independent score coordinates.
    policy_ab = softmax(sp.Matrix([sp.log(PI_REF[i])+scores[i] for i in range(3)]))
    aggregate = sum(w*softplus(-margin(policy_ab,PI_REF,i,j))+l*softplus(margin(policy_ab,PI_REF,i,j))
                    for i,j,w,l in COUNTS)/11
    aggregate = sp.simplify(aggregate)
    for variable in (a,b):
        c.equal(f'aggregate DPO derivative {variable}', sp.diff(aggregate,variable).subs(fitted))
    c.equal('aggregate DPO/reward fit', aggregate.subs(fitted), likelihood.subs(fitted))
    c.equal('individual slope at fitted gap', sp.diff(softplus(-a),a).subs(fitted), -R(1,3))
    report['C_implicit_scores'] = list(implicit.applyfunc(exact))
    report['C_single_observation_loss'] = exact(single)
    report['C_mixed_label_loss'] = exact(likelihood.subs(fitted))
    report['C_fitted_probabilities'] = [exact(sigmoid(margin(optimum, PI_REF, i, j)))
                                      for i, j, _, _ in COUNTS]
    report['C_aggregate_gradient'] = [exact(sp.diff(aggregate, variable).subs(fitted))
                                    for variable in (a, b)]

    # H-D: six real entries, exact chain rule, and a single deliberately chosen step.
    entries = sp.symbols('w11 w12 w21 w22 w31 w32', real=True)
    w = sp.Matrix(3,2,entries)
    h = w*O_STAR
    m = h[0]-h[1]-sp.log(PI_REF[0]/PI_REF[1])
    gradient = sp.Matrix(3,2,[sp.diff(softplus(-m),value) for value in entries])
    at_reference = dict(zip(entries, W_REF))
    g = gradient.subs(at_reference).applyfunc(exact)
    target_gradient = sp.Matrix([[-R(1,3),-R(5,12)],[R(1,3),R(5,12)],[0,0]])
    c.equal('six matrix gradient entries', g, target_gradient)
    c.equal('representation squared norm', O_STAR.dot(O_STAR), R(41,36))
    eta = 72*sp.log(2)/41
    w_new = W_REF - eta*g
    h_new = (w_new*O_STAR).applyfunc(exact)
    p_new = softmax(h_new)
    c.equal('updated logits', h_new, sp.Matrix([sp.log(6),0,0]))
    c.equal('updated probabilities', p_new, sp.Matrix([R(3,4),R(1,8),R(1,8)]))
    c.equal('updated margin', margin(p_new,PI_REF,0,1), sp.log(4))
    c.equal('updated loss', softplus(-margin(p_new,PI_REF,0,1)), sp.log(R(5,4)))
    c.true('unchanged third logit', h_new[2] == (W_REF*O_STAR)[2])
    c.true('changed third probability', p_new[2] != PI_REF[2])
    report['D_gradient'] = g.tolist()
    report['D_symbolic_gradient'] = gradient.tolist()
    report['D_margin'] = m
    report['D_updated_logits'] = list(h_new)
    report['D_updated_policy'] = list(p_new)
    report['D_updated_margin'] = exact(margin(p_new, PI_REF, 0, 1))
    report['D_updated_loss'] = exact(softplus(-margin(p_new, PI_REF, 0, 1)))

    # H-E: beta/reference sensitivity, limits, missing comparisons and support.
    c.equal('uniform reference optimum', tilted_policy([R(1,3)]*3,REWARDS,1)[1],
            sp.Matrix([R(4,7),R(2,7),R(1,7)]))
    beta = sp.symbols('beta', positive=True)
    beta_policy = tilted_policy(PI_REF,REWARDS,beta)[1]
    sensitivity = {}
    targets = {R(1,2):sp.Matrix([R(48,57),R(8,57),R(1,57)]),
               sp.Integer(1):optimum,
               sp.Integer(2):sp.Matrix([6*sp.sqrt(2),4,sp.sqrt(2)])/(7*sp.sqrt(2)+4)}
    for value, target in targets.items():
        _, policy = tilted_policy(PI_REF,REWARDS,value)
        c.equal(f'beta {value}',policy,target)
        sensitivity[str(value)] = list(policy)
    c.equal('large beta limit',beta_policy.applyfunc(lambda x:sp.limit(x,beta,sp.oo)),PI_REF)
    c.equal('small beta limit',beta_policy.applyfunc(lambda x:sp.limit(x,beta,0,dir='+')),sp.Matrix([1,0,0]))
    tied = tilted_policy(PI_REF,[sp.log(2),sp.log(2),0],beta)[1]
    c.equal('tied beta limit',tied.applyfunc(lambda x:sp.limit(x,beta,0,dir='+')),
            sp.Matrix([R(3,5),R(2,5),0]))
    scale = sp.symbols('scale',positive=True)
    c.equal('joint reward/beta scaling',tilted_policy(PI_REF,scale*REWARDS,scale*beta)[1],beta_policy)
    t = sp.symbols('t',positive=True)  # Additionally t<1.
    family = sp.Matrix([3*t/4,t/4,1-t])
    c.equal('missing comparisons margin',margin(family,PI_REF,0,1),sp.log(2))
    c.equal('missing comparisons probability',sigmoid(margin(family,PI_REF,0,1)),R(2,3))
    c.true('zero support infinite KL',kl([R(1,3)]*3,[R(3,5),R(2,5),0]) == sp.oo)
    c.equal('support boundary zero KL',kl([R(3,5),R(2,5),0],[R(3,5),R(2,5),0]))
    c.equal('candidate to reference KL E3',kl(p_new,PI_REF),sp.log(3)-R(11,8)*sp.log(2))
    c.equal('reverse KL E3',kl(PI_REF,p_new),R(11,6)*sp.log(2)-sp.log(3))
    report['E_sensitivity'] = sensitivity

    # H-F: full/modified scores, inconsistent cycle, and absolute-probability failure.
    current_leaves = complete_response_distribution(CURRENT_TREE)
    reference_leaves = complete_response_distribution(REFERENCE_TREE)
    expected_leaves = {
        ('a', 'eos'): (R(1, 4), R(1, 8)),
        ('a', 'd', 'eos'): (R(1, 4), R(3, 8)),
        ('b', 'c', 'eos'): (R(1, 16), R(1, 8)),
        ('b', 'c', 'd', 'eos'): (R(1, 16), R(1, 24)),
        ('b', 'd', 'eos'): (R(1, 8), R(1, 12)),
        ('d', 'eos'): (R(1, 4), R(1, 4)),
    }
    c.true('complete tree common support',
           set(current_leaves) == set(reference_leaves) == set(expected_leaves))
    c.equal('complete tree leaf count', len(current_leaves), 6)
    c.equal('complete tree terminal horizon', max(map(len, current_leaves)), SEQUENCE_HORIZON)
    c.equal('complete current tree normalization', sum(current_leaves.values()), 1)
    c.equal('complete reference tree normalization', sum(reference_leaves.values()), 1)
    for response, (p_value, q_value) in expected_leaves.items():
        c.equal(f'current tree leaf {response}', current_leaves[response], p_value)
        c.equal(f'reference tree leaf {response}', reference_leaves[response], q_value)
    for response, p_factors, q_factors in zip(DISPLAYED_RESPONSES, CURRENT_TOKENS, REFERENCE_TOKENS):
        c.equal(f'current displayed path {response}', sp.prod(p_factors), current_leaves[response])
        c.equal(f'reference displayed path {response}', sp.prod(q_factors), reference_leaves[response])
    sequence_margins = {}
    for mode,target in [('full',sp.log(4)),('mean',R(5,6)*sp.log(2)),('omit_eos',sp.log(R(4,3)))]:
        ratios = [sequence_score(p,q,mode=mode) for p,q in zip(CURRENT_TOKENS,REFERENCE_TOKENS)]
        observed_margin = exact(ratios[0]-ratios[1])
        c.equal(f'sequence margin {mode}',observed_margin,target)
        sequence_margins[mode] = observed_margin
    c.true('mean differs from full',exact(sequence_margins['mean']-sequence_margins['full']) != 0)
    c.true('omit eos differs from full',exact(sequence_margins['omit_eos']-sequence_margins['full']) != 0)
    c.equal('full sequence loss',softplus(-sequence_margins['full']),sp.log(R(5,4)))
    c.equal('omit eos loss',softplus(-sequence_margins['omit_eos']),sp.log(R(7,4)))
    for factors,target in zip(CURRENT_TOKENS,[R(1,4),R(1,16)]):
        c.equal('current complete response',sp.prod(factors),target)
    for factors in REFERENCE_TOKENS:
        c.equal('reference complete response',sp.prod(factors),R(1,8))
    c.equal('cycle odds', (R(2,3)/(1-R(2,3)))**3,8)
    c.equal('scalar cycle telescopes',(s[0]-s[1])+(s[1]-s[2])+(s[2]-s[0]),0)
    c.equal('separable loss infimum',sp.limit(softplus(-d),d,sp.oo),0)
    p_bar = sp.Matrix([R(2,5),R(1,10),R(1,2)])
    c.equal('pbar margin',margin(p_bar,PI_REF,0,1),sp.log(R(8,3)))
    c.equal('pbar loss',softplus(-margin(p_bar,PI_REF,0,1)),sp.log(R(11,8)))
    c.true('chosen probability falls',p_bar[0]<PI_REF[0])
    c.true('comparison loss falls',sp.log(R(11,8))<sp.log(2))
    report['F_sequence_margins'] = sequence_margins
    report['F_terminal_horizon'] = SEQUENCE_HORIZON
    report['F_complete_response_probabilities'] = [
        {'response': list(y), 'current': current_leaves[y], 'reference': reference_leaves[y]}
        for y in expected_leaves]
    report['checks'] = len(c.names)
    return report


if __name__ == '__main__':
    print('Python:', sys.version.split()[0])
    print('SymPy:', sp.__version__)
    print(json.dumps(run_checks(), indent=2, default=str))
