# # Chapter 18: From an attention vector to a tool-use policy update
# This student handout supplies inputs and six investigations. Complete the task cells, add named assertions, and submit your interpretation. The setup runs as supplied; the investigations are deliberately unsolved. Author solutions are separate.
# Use exact arithmetic, declare 0<p,q<1 and c>=0, report versions, and evaluate floats only after symbolic identities. The implementation was checked with SymPy 1.13.3.

import sys
import sympy as sp
from itertools import product
print("Python", sys.version.split()[0], "SymPy", sp.__version__)
Q, I = sp.Rational, sp.Integer
p, q = sp.symbols("p q", positive=True)  # additional domain: p,q < 1
c = sp.symbols("c", nonnegative=True)
scores = sp.Matrix([I(0), sp.log(2), sp.log(3)])
values = sp.Matrix([[1, 0, 1], [0, 1, 1]])
initial = {p: Q(1, 2), q: Q(3, 4), c: Q(1, 10)}
eta = Q(1, 10)
group_paths = ("D5", "V5", "D4", "V5")
group_rewards = sp.Matrix([1, 1, 0, 1])  # Part C uses c=0 only
chain_rewards = [I(0), I(0), I(1)]
chain_values = [Q(1, 4), Q(1, 2), Q(3, 4), I(0)]
terminal_flags = [False, False, True]
clip_cases = [(Q(3, 2), Q(1, 4)), (Q(3, 2), -Q(3, 4)),
              (Q(1, 2), Q(1, 4)), (Q(1, 2), -Q(3, 4))]
current_binary = [Q(3, 4), Q(1, 4)]
reference_binary = [Q(1, 4), Q(3, 4)]
old_binary = [Q(1, 2), Q(1, 2)]
length_contributions = [[I(1), I(1)], [-I(1)] * 4]
checks = []
def check(name, actual, expected):
    residual = sp.simplify(actual - expected)
    zero = sp.zeros(*residual.shape) if isinstance(residual, sp.MatrixBase) else 0
    if residual != zero:
        raise AssertionError(name)
    checks.append(name)
    print("PASS", name)
check("input.values_shape", sp.Integer(values.rows * 10 + values.cols), I(23))
check("input.group_size", I(len(group_paths)), I(4))

# ## Part A: Reconstruct the inherited attention and the complete policy tree
#
# Build \(s\), the three values, the exact softmax weights, and \(z\). Introduce symbolic \(p,q,c\), construct all four complete-path probabilities and utilities, and verify normalization and the closed form of \(J\). Substitute \(p=1/2,q=3/4,c=1/10\) and calculate correctness, expected tool count, and utility separately.

# Part A: derive, verify with named assertions, and explain assumptions.

# ## Part B: Derive and apply an exact expected-gradient update
#
# Compute \(\partial_\alpha J\) and \(\partial_\phi J\) using the sigmoid chain rule, then compute them independently from the four path scores. Map the initial-logit derivative to \(\nabla_wJ\). Verify the constant-baseline identity symbolically. Apply the declared ascent step \(\eta=1/10\) to \(w\) and \(\phi\), recompute probabilities and utility, and label the result as an expected-gradient toy update rather than a sampled optimizer result.

# Part B: derive, verify with named assertions, and explain assumptions.

# ## Part C: Compare group baselines and verify their expectations
#
# For the **outcome-only, \(c=0\)** group specified in the supplied local inputs, calculate centered, standardized, and LOO coefficients and the resulting sample gradients. Implement an explicit zero-variance rule. Enumerate all \(4^4=256\) ordered groups from the complete tree at the initial probabilities, multiplying leaf probabilities to obtain each group’s probability. Verify the expected self-including estimate and the LOO estimate against the exact outcome-only gradient. Do not use Monte Carlo sampling to establish an identity that can be enumerated exactly.

# Part C: derive, verify with named assertions, and explain assumptions.

# ## Part D: Make clipping, KL, and reduction choices explicit
#
# Implement the scalar PPO surrogate using exact arithmetic and a piecewise representation or exact `Min`/`Max`. Verify all four clipping cases. Compute the exact binary KL for current \(3/4\) and reference \(1/4\). For \(k=u-\log u-1\), compare its expectation under the current action distribution with its average under the old \((1/2,1/2)\) distribution. Finally, implement the three reduction conventions for the two contribution sequences in the supplied local inputs.

# Part D: derive, verify with named assertions, and explain assumptions.

# ## Part E: Audit ownership and off-policy correction
#
# Represent the verification path as typed events with ownership mask \((1,0,1)\). Construct its policy likelihood and show the extra factor caused by treating the deterministic observation as another \(q\)-head action. Using behavior \(p_\mu=q_\mu=1/2\), compute full trajectory importance weights and recover the target utility and logit gradients by exact summation. Implement a support check that rejects the missing-support example rather than dividing by zero or silently dropping the unavailable target action.

# Part E: derive, verify with named assertions, and explain assumptions.

# ## Part F: Trace delayed reward, cutoffs, and shaping boundaries
#
# Compute the supplementary chain’s returns, TD residuals, and GAE for \(\lambda=1/2\) and \(\lambda=1\). Implement separate terminal and truncation flags. Calculate the step-2 target for administrative truncation and true termination. Symbolically verify the potential-shaping telescoping identity, including its terminal term. Demonstrate fragment-additive log scores and exhibit a changed result when fragment means are weighted independently. End with a short explanation of which conclusions are exact, which depend on value estimates, and which would require empirical evidence in a real agent.

# Part F: derive, verify with named assertions, and explain assumptions.

# ## Submission
# Run from a clean kernel, add a visible pass/fail summary for your completed assertions, and distinguish exact results, supplied value estimates, and conclusions needing empirical evidence. Suggested grading: A10%, B25%, C20%, D15%, E15%, F15%.
