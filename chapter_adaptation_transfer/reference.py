"""Extended adaptation fixtures for merged Chapter 17.

The current A--D homework reference is homework_reference.py."""
from __future__ import annotations
import sympy as sp

R = sp.Rational
checks: list[str] = []

def clean(expr: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.expand_log(expr))

def eq(name: str, actual, expected) -> None:
    """Require exact equality; never silently fall back to numerical tolerance."""
    if isinstance(actual, sp.MatrixBase) or isinstance(expected, sp.MatrixBase):
        a, b = sp.Matrix(actual), sp.Matrix(expected)
        if a.shape != b.shape:
            raise AssertionError(f"{name}: shape mismatch {a.shape} != {b.shape}")
        residuals = [clean(v) for v in a - b]
    else:
        residuals = [clean(sp.sympify(actual) - sp.sympify(expected))]
    if any(v != 0 for v in residuals):
        raise AssertionError(f"{name}: residuals {residuals}")
    checks.append(name)

def softmax(logits, tau=sp.S.One) -> sp.Matrix:
    e = sp.Matrix([sp.exp(x / tau) for x in logits])
    return e / sum(e)

def kl(averaging_law, predictor) -> sp.Expr:
    """Interior-simplex KL only; boundary cases are taken as explicit limits."""
    if len(averaging_law) != len(predictor):
        raise ValueError("KL requires the same ordered event space")
    return sum(pi * sp.log(pi / qi) for pi, qi in zip(averaging_law, predictor))

# A. Inherited attention and explicitly new readout.
a = sp.Matrix([R(1, 6), R(1, 3), R(1, 2)])
V = sp.Matrix([[1, 0], [0, 1], [1, 1]])
o_star = V.T * a
W_base = sp.Matrix([[1, 0], [1, 0], [1, 0]])
p_teacher = sp.Matrix([R(1, 2), R(1, 3), R(1, 6)])
p_student0 = sp.ones(3, 1) / 3
teacher_logits = sp.Matrix([sp.log(3), sp.log(2), 0])
eq("attention normalization", sum(a), 1)
eq("attention weights from scores", softmax([0, sp.log(2), sp.log(3)]), a)
eq("inherited attention output", o_star, sp.Matrix([R(2, 3), R(5, 6)]))
eq("readout logits", W_base * o_star, sp.ones(3, 1) * R(2, 3))
eq("readout uniform distribution", softmax(W_base * o_star), p_student0)
eq("teacher distribution", softmax(teacher_logits), p_teacher)
eq("teacher normalization", sum(p_teacher), 1)
eq("token-normalized SFT", (sp.log(2) + 3 * sp.log(4)) / 4, R(7, 4) * sp.log(2))
eq("record-normalized SFT", (sp.log(2) + sp.log(4)) / 2, R(3, 2) * sp.log(2))

# B. Low-rank parameterization, general differential, initialization, and merge.
A0 = sp.Matrix([[1, 0]])
B0 = sp.zeros(3, 1)
g = p_student0 - p_teacher
G = g * o_star.T
dB = G * A0.T
dA = B0.T * G
eq("initial low-rank update", B0 * A0, sp.zeros(3, 2))
eq("soft-target logit gradient", g, sp.Matrix([-R(1, 6), 0, R(1, 6)]))
eq("hard-target logit gradient", p_student0 - sp.eye(3)[:, 0], sp.Matrix([-R(2, 3), R(1, 3), R(1, 3)]))
expected_sampled_gradient = sum((p_teacher[i] * (p_student0 - sp.eye(3)[:, i]) for i in range(3)), sp.zeros(3, 1))
eq("sampled-teacher expected gradient", expected_sampled_gradient, g)
eq("effective-weight gradient", G, sp.Matrix([[-R(1, 9), -R(5, 36)], [0, 0], [R(1, 9), R(5, 36)]]))
eq("first B gradient", dB, sp.Matrix([-R(1, 9), 0, R(1, 9)]))
eq("first A gradient", dA, sp.zeros(1, 2))
a1, a2, b1, b2, b3 = sp.symbols("a1 a2 b1 b2 b3", real=True)
gamma = sp.symbols("gamma", positive=True)
A = sp.Matrix([[a1, a2]])
B = sp.Matrix([b1, b2, b3])
GG = sp.Matrix(3, 2, sp.symbols("g0:6", real=True))
linear_loss = sum(GG[i, j] * (gamma * B * A)[i, j] for i in range(3) for j in range(2))
eq("general B chain rule", sp.Matrix([sp.diff(linear_loss, x) for x in B]), gamma * GG * A.T)
eq("general A chain rule", sp.Matrix([[sp.diff(linear_loss, x) for x in A]]), gamma * B.T * GG)
z_student = (W_base + B * A) * o_star
ce = sp.log(sum(sp.exp(x) for x in z_student)) - (p_teacher.T * z_student)[0]
init = {a1: 1, a2: 0, b1: 0, b2: 0, b3: 0}
zeros = {a1: 0, a2: 0, b1: 0, b2: 0, b3: 0}
eq("direct CE differentiation B", sp.Matrix([sp.diff(ce, x).subs(init) for x in B]), dB)
eq("direct CE differentiation A", sp.Matrix([[sp.diff(ce, x).subs(init) for x in A]]), dA)
eq("both-zero B gradient", sp.Matrix([sp.diff(ce, x).subs(zeros) for x in B]), sp.zeros(3, 1))
eq("both-zero A gradient", sp.Matrix([[sp.diff(ce, x).subs(zeros) for x in A]]), sp.zeros(1, 2))
Bnew = B0 - dB
Anew = A0 - dA
eq("one-step B", Bnew, sp.Matrix([R(1, 9), 0, -R(1, 9)]))
eq("one-step update rank", (Bnew * Anew).rank(), 1)
eq("merged forward identity", (W_base + Bnew * Anew) * o_star, W_base * o_star + Bnew * (Anew * o_star))
eq("one-step logits", (W_base + Bnew * Anew) * o_star, sp.Matrix([R(20, 27), R(18, 27), R(16, 27)]))
eq("factor gauge invariance", (2 * Bnew) * (Anew / 2), Bnew * Anew)
eq("non-subspace counterexample", (sp.diag(1, 0) + sp.diag(0, 1)).rank(), 2)
eq("full rank need not be lost", (sp.eye(2) + sp.diag(1, 0)).rank(), 2)
eq("stored adapter scalars", 8 * (4096 + 4096), 65536)
eq("full matrix scalars", 4096 * 4096, 16777216)
eq("parameter count ratio", R(4096 * 4096, 8 * (4096 + 4096)), 256)

# C. Temperature and both directions of categorical KL.
x = sp.Matrix(sp.symbols("x0:3", real=True))
tau = sp.symbols("tau", positive=True)
qt = softmax(x, tau)
pt = softmax(teacher_logits, tau)
kd = tau**2 * kl(pt, qt)
for j in range(3):
    eq(f"temperature-scaled gradient {j}", sp.diff(kd, x[j]), tau * (qt[j] - pt[j]))
eq("temperature-two teacher", pt.subs(tau, 2), sp.Matrix([sp.sqrt(3), sp.sqrt(2), 1]) / (sp.sqrt(3) + sp.sqrt(2) + 1))
p_student = softmax(x)
DF = kl(p_teacher, p_student)
DR = kl(p_student, p_teacher)
for j in range(3):
    eq(f"forward logit gradient {j}", sp.diff(DF, x[j]), p_student[j] - p_teacher[j])
    eq(f"reverse logit gradient {j}", sp.diff(DR, x[j]), p_student[j] * (sp.log(p_student[j] / p_teacher[j]) - DR))
    expected_score = sum(p_student[i] * sp.log(p_student[i] / p_teacher[i]) * sp.diff(sp.log(p_student[i]), x[j]) for i in range(3))
    eq(f"fixed-prefix unclipped score identity {j}", expected_score, sp.diff(DR, x[j]))
eq("forward fixture KL", kl(p_teacher, p_student0), sp.log(3) / 2 - R(2, 3) * sp.log(2))
eq("reverse fixture KL", kl(p_student0, p_teacher), sp.log(R(4, 3)) / 3)
eq("cross-entropy entropy decomposition", -sum(pi * sp.log(qi) for pi, qi in zip(p_teacher, p_student0)), -sum(pi * sp.log(pi) for pi in p_teacher) + kl(p_teacher, p_student0))
eps = sp.symbols("eps", positive=True)
peps = sp.Matrix([1 - eps, eps])
half = sp.Matrix([R(1, 2), R(1, 2)])
eq("forward boundary limit", sp.limit(kl(peps, half), eps, 0, dir="+"), sp.log(2))
reverse_boundary = sp.limit(kl(half, peps), eps, 0, dir="+")
if reverse_boundary != sp.oo:
    raise AssertionError("reverse boundary must be +infinity")
checks.append("reverse boundary limit")
# Taylor coefficients at inverse temperature zero, for a nontrivial fixed-logit fixture.
invT = sp.symbols("invT", real=True)
slog = sp.Matrix([0, sp.log(2), sp.log(4)])
p_eps = softmax(teacher_logits * invT)
q_eps = softmax(slog * invT)
centered = (slog - sp.ones(3, 1) * sum(slog) / 3) - (teacher_logits - sp.ones(3, 1) * sum(teacher_logits) / 3)
for j in range(3):
    eq(f"high-temperature gradient coefficient {j}", sp.diff(q_eps[j] - p_eps[j], invT).subs(invT, 0), centered[j] / 3)
eq("high-temperature centered-square coefficient", sp.diff(kl(p_eps, q_eps), invT, 2).subs(invT, 0) / 2, (centered.T * centered)[0] / 6)

# D. Two fixed-history teachers: arithmetic versus normalized geometric mean.
p2 = sp.Matrix([R(1, 6), R(1, 3), R(1, 2)])
pbar = (p_teacher + p2) / 2
geometric_raw = sp.Matrix([sp.sqrt(p_teacher[i] * p2[i]) for i in range(3)])
Z = sum(geometric_raw)
pgeo = geometric_raw / Z
eq("arithmetic teacher", pbar, p_student0)
eq("geometric normalization", sum(pgeo), 1)
eq("geometric normalizer", Z, (sp.sqrt(3) + 1) / 3)
eq("geometric middle coordinate", pgeo[1], 1 / (sp.sqrt(3) + 1))
v1, v2 = sp.symbols("v1 v2", positive=True)
trial_q = sp.Matrix([v1, v2, 1]) / (v1 + v2 + 1)
eq("forward teacher barycenter identity", (kl(p_teacher, trial_q) + kl(p2, trial_q)) / 2, kl(pbar, trial_q) + (kl(p_teacher, pbar) + kl(p2, pbar)) / 2)
eq("reverse teacher barycenter identity", (kl(trial_q, p_teacher) + kl(trial_q, p2)) / 2, kl(trial_q, pgeo) - sp.log(Z))

# E. Supplementary two-step binary tree, with u in (0,1).
u = sp.symbols("u", positive=True)  # Additional declared domain condition: u < 1.
root_p = sp.Matrix([R(3, 4), R(1, 4)])
root_q = sp.Matrix([u, 1 - u])
a_p = half
b_p = root_p
P = sp.Matrix([R(3, 8), R(3, 8), R(3, 16), R(1, 16)])
Q = sp.Matrix([u / 2, u / 2, (1 - u) / 2, (1 - u) / 2])
c = kl(half, b_p)
f = kl(root_p, half)
eq("teacher joint normalization", sum(P), 1)
eq("student joint normalization", sum(Q), 1)
eq("teacher joint factorization", P, sp.Matrix([root_p[0] * a_p[0], root_p[0] * a_p[1], root_p[1] * b_p[0], root_p[1] * b_p[1]]))
eq("forward sequence chain rule", kl(P, Q), kl(root_p, root_q) + root_p[0] * kl(a_p, half) + root_p[1] * kl(b_p, half))
eq("reverse sequence chain rule", kl(Q, P), kl(root_q, root_p) + u * kl(half, a_p) + (1 - u) * kl(half, b_p))
eq("conditional reverse cost", c, sp.log(R(4, 3)) / 2)
eq("forward sequence fixture", kl(P, Q).subs(u, R(1, 2)), R(5, 4) * f)
eq("reverse sequence fixture", kl(Q, P).subs(u, R(1, 2)), R(3, 2) * c)
J = kl(root_q, root_p) + (1 - u) * c
J_frozen = kl(root_q, root_p) + R(1, 2) * c
eq("matching objective values at snapshot", J.subs(u, R(1, 2)), J_frozen.subs(u, R(1, 2)))
eq("missing occupancy derivative", sp.diff(J, u) - sp.diff(J_frozen, u), -c)
eq("frozen derivative at snapshot", sp.diff(J_frozen, u).subs(u, R(1, 2)), -sp.log(3))
eq("true derivative at snapshot", sp.diff(J, u).subs(u, R(1, 2)), -sp.log(3) - c)
eq("informative prefix frequency ratio", (1 - R(1, 2)) / R(1, 4), 2)

if __name__ == "__main__":
    print(f"SymPy {sp.__version__}: {len(checks)} named exact checks passed.")
    for number, name in enumerate(checks, 1):
        print(f"{number:02d}. PASS {name}")
