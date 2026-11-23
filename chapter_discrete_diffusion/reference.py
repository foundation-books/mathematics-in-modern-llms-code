"""Exact CPU laboratory for discrete diffusion; no training or timing claims.

All probability vectors are rows. Pair states use lexicographic A,B,C,M
order. Unsupported Bayes rows are tagged and filled with identity rows;
their values are arbitrary and are never claimed to be data posteriors.
"""

from itertools import product
import json
import sys

import sympy as sp

R = sp.Rational
N_vocab = 3
MASK = N_vocab
BETAS = (R(1, 2), R(1, 2), sp.Integer(1))
ALPHAS = (sp.Integer(1), R(1, 2), R(1, 4), sp.Integer(0))
RHO = sp.Matrix([[R(4, 15), R(1, 3), R(2, 5)]])
STATES = tuple(product(range(N_vocab + 1), repeat=2))
INDEX = {state: i for i, state in enumerate(STATES)}


def check_stochastic(matrix):
    """Reject nonnumeric/negative entries, nonsquare kernels, or lost mass."""
    if matrix.rows != matrix.cols:
        raise ValueError("kernel must be square")
    if any(v.is_nonnegative is not True for v in matrix):
        raise ValueError("kernel has a negative or unresolved entry")
    if matrix * sp.ones(matrix.cols, 1) != sp.ones(matrix.rows, 1):
        raise ValueError("kernel row mass is not one")


def checked_beta(beta):
    """A replacement/masking probability must lie in [0,1]."""
    beta = sp.sympify(beta)
    if beta.is_nonnegative is not True or (1 - beta).is_nonnegative is not True:
        raise ValueError("noise probability must lie in [0,1]")
    return beta


def uniform_kernel(beta, size=N_vocab):
    beta = checked_beta(beta)
    matrix = (1 - beta) * sp.eye(size) + beta * sp.ones(size) / size
    check_stochastic(matrix)
    return matrix


def mask_kernel(beta, size=N_vocab):
    beta = checked_beta(beta)
    matrix = sp.eye(size + 1)
    for i in range(size):
        matrix[i, i] = 1 - beta
        matrix[i, size] = beta
    check_stochastic(matrix)
    return matrix


def cumulative(kernels):
    """Return I followed by cumulative row-convention kernel products."""
    if not kernels:
        raise ValueError("at least one kernel is required")
    result = [sp.eye(kernels[0].rows)]
    for kernel in kernels:
        check_stochastic(kernel)
        result.append(result[-1] * kernel)
    return result


def posterior(previous, step, clean, current):
    """B(i|current,clean), rejecting an unsupported conditioning event."""
    if previous.shape != step.shape:
        raise ValueError("kernel dimensions disagree")
    denominator = (previous * step)[clean, current]
    if denominator.is_positive is not True:
        raise ValueError("conditioning event has zero or unresolved probability")
    return sp.Matrix([[
        previous[clean, i] * step[i, current] / denominator
        for i in range(step.rows)
    ]])


def reverse_from_marginal(previous_law, step):
    """Rows: current state, columns: previous state; return support flags."""
    check_stochastic(step)
    if previous_law.shape != (1, step.rows):
        raise ValueError("law must be a row matching the kernel")
    current_law = previous_law * step
    reverse = sp.zeros(step.rows)
    supported = []
    for j in range(step.rows):
        mass = current_law[j]
        supported.append(mass > 0)
        if mass == 0:
            # Arbitrary normalized extension. Multiplied by zero data mass.
            reverse[j, j] = 1
        else:
            for i in range(step.rows):
                reverse[j, i] = previous_law[i] * step[i, j] / mass
    check_stochastic(reverse)
    return reverse, tuple(supported)


def kl(q, p):
    """KL with 0 log(0/p)=0 and positive mass/zero model mass = infinity."""
    if len(q) != len(p):
        raise ValueError("probability vector lengths disagree")
    result = sp.Integer(0)
    for qi, pi in zip(q, p):
        if qi == 0:
            continue
        if pi == 0:
            return sp.oo
        result += qi * sp.log(qi / pi)
    return sp.simplify(result)


def reveal_weight(alpha_previous, alpha_current):
    """Impossible-mask event returns None, not an evaluated 0/0."""
    if not (0 <= alpha_current <= alpha_previous <= 1):
        raise ValueError("retention must decrease within [0,1]")
    if alpha_current == 1:
        return None
    return sp.cancel((alpha_previous - alpha_current) / (1 - alpha_current))


WEIGHTS = tuple(reveal_weight(ALPHAS[t - 1], ALPHAS[t]) for t in range(1, 4))


def pair_product_reverse(weight):
    """Synchronous rule: construct both distributions from the old tuple."""
    matrix = sp.zeros(len(STATES))
    for old_index, old in enumerate(STATES):
        distributions = []
        for position in range(2):
            token, other = old[position], old[1 - position]
            row = sp.zeros(1, N_vocab + 1)
            if token != MASK:
                row[token] = 1
            else:
                clean = RHO if other == MASK else sp.eye(N_vocab)[other, :]
                for k in range(N_vocab):
                    row[k] = weight * clean[k]
                row[MASK] = 1 - weight
            distributions.append(row)
        for new_index, new in enumerate(STATES):
            matrix[old_index, new_index] = (
                distributions[0][new[0]] * distributions[1][new[1]]
            )
    check_stochastic(matrix)
    return matrix


def unequal_mass(law):
    """Unequal *clean* pairs; residual mask mass is checked separately."""
    return sum(law[i] for i, (a, b) in enumerate(STATES)
               if a < N_vocab and b < N_vocab and a != b)


def pair_laboratory():
    clean = sp.zeros(1, len(STATES))
    for k in range(N_vocab):
        clean[INDEX[k, k]] = RHO[k]
    forward = [sp.kronecker_product(mask_kernel(b), mask_kernel(b))
               for b in BETAS]
    marginals = [clean]
    reverses, support = [], []
    for step in forward:
        reverse, supported = reverse_from_marginal(marginals[-1], step)
        reverses.append(reverse)
        support.append(supported)
        marginals.append(marginals[-1] * step)
    exact = marginals[-1]
    factorized = marginals[-1]
    exact_trace, factorized_trace = [exact], [factorized]
    for t in range(3, 0, -1):
        exact = exact * reverses[t - 1]
        factorized = factorized * pair_product_reverse(WEIGHTS[t - 1])
        exact_trace.append(exact)
        factorized_trace.append(factorized)
    one_round = sp.kronecker_product(RHO.row_join(sp.zeros(1, 1)),
                                     RHO.row_join(sp.zeros(1, 1)))
    return {
        "forward": forward, "marginals": marginals, "reverses": reverses,
        "support": support, "exact_trace": exact_trace,
        "factorized_trace": factorized_trace, "one_round": one_round,
    }


def cost_ledger(denoise_cost=sp.Integer(2)):
    evaluations, commits, width, valid = 7, 2, 4, 12
    calls = evaluations + commits
    latency = evaluations * denoise_cost + commits * R(1, 2)
    return {
        "denoise": evaluations, "commits": commits,
        "generation_calls": calls,
        "valid_tokens_per_generation_call": R(valid, calls),
        "valid_tokens_per_denoising_evaluation": R(valid, evaluations),
        "processed_positions": width * calls,
        "diffusion_latency": latency, "ar_latency": sp.Integer(valid),
        "ar_over_diffusion": valid / latency,
        "strict_denoise_cost_limit": R(11, 7),
    }


def entropy(probabilities):
    return -sum(p * sp.log(p) for p in probabilities if p != 0)


def should_stop(round_index, current_prediction, previous_prediction,
                mean_entropy, threshold, cap):
    """Illustrative bounded rule; caller returns current greedy prediction."""
    if cap < 1 or not 1 <= round_index <= cap:
        raise ValueError("round must lie within a positive cap")
    if round_index == cap:
        return True
    return (round_index >= 2 and previous_prediction is not None
            and current_prediction == previous_prediction
            and mean_entropy <= threshold)


def verify():
    """Independent fixture assertions plus exhaustive finite state checks."""
    checks = 0

    def expect(condition):
        nonlocal checks
        if not condition:
            raise AssertionError(f"check {checks + 1} failed")
        checks += 1

    o_star = sp.Matrix([R(2, 3), R(5, 6)])
    raw = sp.Matrix([[o_star[0], o_star[1], 1]])
    expect(raw / sum(raw) == RHO)
    uniform = [uniform_kernel(b) for b in BETAS]
    masked = [mask_kernel(b) for b in BETAS]
    uq, mq = cumulative(uniform), cumulative(masked)
    u = sp.ones(N_vocab) / N_vocab
    expect(u * u == u)
    for t in range(4):
        expect(uq[t] == ALPHAS[t] * sp.eye(N_vocab) + (1 - ALPHAS[t]) * u)
        expect(mq[t] == mask_kernel(1 - ALPHAS[t]))
    expect(RHO * uq[1] == sp.Matrix([[R(3, 10), R(1, 3), R(11, 30)]]))
    expect(RHO * uq[2] == sp.Matrix([[R(19, 60), R(1, 3), R(7, 20)]]))
    expect(RHO * uq[3] == sp.ones(1, N_vocab) / N_vocab)
    rho_plus = RHO.row_join(sp.zeros(1, 1))
    expect(rho_plus * mq[2] == sp.Matrix([[R(1, 15), R(1, 12), R(1, 10), R(3, 4)]]))
    expect(rho_plus * mq[3] == sp.Matrix([[0, 0, 0, 1]]))
    expect(4 * (1 - ALPHAS[2]) == 3)
    expect(ALPHAS[2] ** 4 == R(1, 256))
    expect(BETAS[0] * (1 - R(1, N_vocab)) == R(1, 3))
    expect(posterior(uq[1], uniform[1], 0, 1) == sp.Matrix([[R(4, 9), R(4, 9), R(1, 9)]]))
    expect(posterior(mq[1], masked[1], 0, MASK) == sp.Matrix([[R(1, 3), 0, 0, R(2, 3)]]))
    try:
        posterior(mq[1], masked[1], 0, 1)
    except ValueError:
        expect(True)
    else:
        raise AssertionError("impossible posterior was accepted")
    clean_posterior = sp.Matrix([[RHO[k] * uq[2][k, 1] for k in range(N_vocab)]])
    clean_posterior /= sum(clean_posterior)
    expect(clean_posterior == sp.Matrix([[R(1, 5), R(1, 2), R(3, 10)]]))
    mixture = sp.zeros(1, N_vocab)
    for k in range(N_vocab):
        mixture += clean_posterior[k] * posterior(uq[1], uniform[1], k, 1)
    direct, _ = reverse_from_marginal(RHO * uq[1], uniform[1])
    expect(mixture == direct[1, :])
    expect(mixture == sp.Matrix([[R(3, 20), R(2, 3), R(11, 60)]]))
    expect(WEIGHTS == (1, R(1, 3), R(1, 4)))
    w, dk = sp.symbols("w dk", positive=True)
    # The mask ratio equals one. Its term is zero also when w=1.
    expect(sp.simplify(w * sp.log(w / (w * dk)) + w * sp.log(dk)) == 0)
    logits = sp.symbols("z0:3", real=True)
    normalizer = sum(sp.exp(l) for l in logits)
    loss = R(1, 3) * (sp.log(normalizer) - logits[0])
    probabilities = [sp.exp(l) / normalizer for l in logits]
    for i, l in enumerate(logits):
        expect(sp.simplify(sp.diff(loss, l) - R(1, 3) * (probabilities[i] - int(i == 0))) == 0)
    substitution = dict(zip(logits, (sp.log(R(2, 3)), sp.log(R(5, 6)), 0)))
    gradient = sp.Matrix([sp.simplify(sp.diff(loss, l).subs(substitution)) for l in logits])
    expect(gradient == sp.Matrix([R(-11, 45), R(1, 9), R(2, 15)]))
    q2 = sp.Matrix([[R(1, 3), 0, 0, R(2, 3)]])
    p2 = sp.Matrix([[R(4, 45), R(1, 9), R(2, 15), R(2, 3)]])
    expect(sp.simplify(kl(q2, p2) - R(1, 3) * sp.log(R(15, 4))) == 0)
    coefficients = [WEIGHTS[t - 1] * (1 - ALPHAS[t]) for t in range(1, 4)]
    nu = (R(1, 2), R(1, 3), R(1, 6))
    expect(sum(coefficients) == 1)
    expect(sum(nu[i] * coefficients[i] / nu[i] for i in range(3)) == 1)
    expect(sum(nu[i] * coefficients[i] for i in range(3)) == R(3, 8))
    # Independently enumerate the finite one-token path likelihood/bound.
    model_steps = []
    for weight in WEIGHTS:
        reverse = sp.eye(N_vocab + 1)
        reverse[MASK, :] = (weight * rho_plus
                            + (1 - weight) * sp.Matrix([[0, 0, 0, 1]]))
        model_steps.append(reverse)
    path_likelihood, path_bound = sp.Integer(0), sp.Integer(0)
    for x1, x2, x3 in product(range(N_vocab + 1), repeat=3):
        qpath = masked[0][0, x1] * masked[1][x1, x2] * masked[2][x2, x3]
        ppath = (int(x3 == MASK) * model_steps[2][x3, x2]
                 * model_steps[1][x2, x1] * model_steps[0][x1, 0])
        path_likelihood += ppath
        if qpath != 0:
            path_bound += qpath * sp.log(qpath / ppath)
    expect(path_likelihood == RHO[0])
    expect(sp.simplify(path_bound - sp.log(R(15, 4))) == 0)
    # A four-token trace is possible, but mask-count expectations are not quotas.
    trace_probability = (WEIGHTS[2] * RHO[0] * (1 - WEIGHTS[2]) ** 2
                         * WEIGHTS[2] * RHO[2]
                         * WEIGHTS[1] * RHO[1] * (1 - WEIGHTS[1]) * RHO[1])
    expect(trace_probability == R(1, 10800))
    expect(kl([R(1, 4), R(3, 4)], [0, 1]) == sp.oo)
    expect(kl([0, 1], [0, 1]) == 0)
    pair = pair_laboratory()
    for matrix in pair["forward"] + pair["reverses"]:
        for row in range(16):
            expect(sum(matrix[row, :]) == 1)
            expect(all(v >= 0 for v in matrix[row, :]))
    for marginal in pair["marginals"] + pair["exact_trace"] + pair["factorized_trace"]:
        expect(sum(marginal) == 1)
        expect(all(v >= 0 for v in marginal))
    for t in range(1, 4):
        expect(pair["marginals"][t] * pair["reverses"][t - 1] == pair["marginals"][t - 1])
        product_reverse = pair_product_reverse(WEIGHTS[t - 1])
        # All supported noisy states: check both reverse position marginals.
        for j, is_supported in enumerate(pair["support"][t - 1]):
            if not is_supported:
                expect(pair["marginals"][t][j] == 0)
                continue
            for position in range(2):
                for token in range(4):
                    exact_mass = sum(pair["reverses"][t - 1][j, i]
                                     for i, state in enumerate(STATES) if state[position] == token)
                    product_mass = sum(product_reverse[j, i]
                                       for i, state in enumerate(STATES) if state[position] == token)
                    expect(exact_mass == product_mass)
    exact_final, product_final = pair["exact_trace"][-1], pair["factorized_trace"][-1]
    expect(exact_final == pair["marginals"][0])
    expect(unequal_mass(exact_final) == 0)
    expect(unequal_mass(pair["one_round"]) == R(148, 225))
    expect(unequal_mass(product_final) == R(37, 150))
    expect(sum(product_final[i] for i, state in enumerate(STATES) if MASK in state) == 0)
    expect((1 - sum(p ** 2 for p in RHO)) * sum((ALPHAS[t - 1] - ALPHAS[t]) ** 2 for t in range(1, 4)) == R(37, 150))
    ledger = cost_ledger()
    expect(ledger["generation_calls"] == 9)
    expect(ledger["valid_tokens_per_generation_call"] == R(4, 3))
    expect(ledger["valid_tokens_per_denoising_evaluation"] == R(12, 7))
    expect(ledger["processed_positions"] == 36)
    expect(ledger["diffusion_latency"] == 15)
    expect(ledger["ar_over_diffusion"] == R(4, 5))
    expect(cost_ledger(R(11, 7))["diffusion_latency"] == 12)
    expect(cost_ledger(sp.Integer(1))["ar_over_diffusion"] == R(3, 2))
    mean_entropy = sp.log(3) / 256
    expect(mean_entropy.evalf(50) < R(5, 1000))
    expect(entropy([R(1, 3)] * 3) == sp.log(3))
    expect(entropy([0, 1, 0]) == 0)
    expect(should_stop(2, (0,) * 256, (0,) * 256, mean_entropy, R(5, 1000), 48))
    return {
        "python": sys.version.split()[0], "sympy": sp.__version__,
        "assertions": checks, "joint_states": len(STATES),
        "reverse_steps": 3, "unequal_exact": str(unequal_mass(exact_final)),
        "unequal_one_round": str(unequal_mass(pair["one_round"])),
        "unequal_three_round": str(unequal_mass(product_final)),
        "cost": {k: str(v) for k, v in ledger.items()},
        "mean_entropy": str(mean_entropy),
        "mean_entropy_decimal": str(mean_entropy.evalf(12)),
        "nonzero_product_final": {str(state): str(product_final[i])
                                  for i, state in enumerate(STATES) if product_final[i] != 0},
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
