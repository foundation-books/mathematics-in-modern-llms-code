"""Chapter 14's CPU-only mathematical laboratory; no training benchmark.

Exact real fixtures use SymPy rationals. Longer polynomial runs use 80 digits.
The quintic experiments share the cubic normalization and do not emulate BF16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys

import sympy as sp

q = sp.Rational
O_STAR = sp.Matrix([q(2, 3), q(5, 6)])
INPUT_FEATURE2 = sp.Matrix([0, 1])
READOUT = sp.Matrix([[1, 0], [0, 1], [0, 0], [0, 0]])
W0 = sp.zeros(2)
G1 = sp.Matrix([[q(1, 6), q(5, 24)], [-q(1, 2), -q(5, 8)]])
G2 = sp.Matrix([[0, -q(3, 4)], [0, q(1, 4)]])
GBAR = (G1 + G2) / 2
M = sp.Matrix([[3, 1], [1, 3]])
A = sp.Matrix([[3, 0, 0], [0, 1, 0]])
A0 = sp.Matrix([[3, 0, 0], [0, 0, 0]])
MH = sp.Matrix([[3, 0], [1, 1]])
Q1 = sp.Matrix([[4, 5], [-12, -15]]) / sp.sqrt(410)
QBAR = sp.Matrix([[13, -25], [-25, -13]]) / sp.sqrt(794)
DA = sp.Matrix([[q(5, 11), -q(65, 89)], [-q(5, 7), -q(15, 23)]])
PRECISION = 80
AGREEMENT_TOLERANCE = sp.Rational(1, 10**30)
COUNTS = (0, 1, 2, 5, 10)
QUINTIC = (q(6889, 2000), -q(191, 40), q(4063, 2000))


def simplify_matrix(mat):
    return sp.Matrix(mat).applyfunc(sp.simplify)


def equal(left, right):
    if isinstance(left, sp.MatrixBase) or isinstance(right, sp.MatrixBase):
        left, right = sp.Matrix(left), sp.Matrix(right)
        return left.shape == right.shape and all(sp.simplify(x) == 0 for x in left - right)
    return sp.simplify(left - right) == 0


def frob(mat):
    """Real-matrix Frobenius norm (all inputs here are real)."""
    return sp.sqrt(sum(x*x for x in mat))


def rms(mat):
    return frob(mat) / sp.sqrt(mat.rows * mat.cols)


def token_loss(weights, input_feature, target):
    logits = READOUT * weights * input_feature
    return sp.log(sum(sp.exp(x) for x in logits)) - logits[target]


def symbolic_gradients():
    entries = sp.symbols('w11 w12 w21 w22', real=True)
    weights = sp.Matrix(2, 2, entries)
    zero_subs = dict.fromkeys(entries, 0)
    losses = (token_loss(weights, O_STAR, 1), token_loss(weights, INPUT_FEATURE2, 0))
    grads = [sp.Matrix(2, 2, [sp.diff(loss, x) for x in entries]) for loss in losses]
    at_zero = [simplify_matrix(g.subs(zero_subs)) for g in grads]
    # Differentiate the mean separately; do not only average constants.
    mean_loss = sum(losses) / 2
    mean_grad = sp.Matrix(2, 2, [sp.diff(mean_loss, x) for x in entries])
    return losses, grads, at_zero, simplify_matrix(mean_grad.subs(zero_subs))


def supported_polar(mat):
    """Exact supported factor for small real fixtures, including rectangular rank loss.

    A rank-one factor has one singular value ||M||F. Invertible real 2x2
    matrices use the proved rotation/reflection certificate. Other small exact
    matrices use SymPy's condensed SVD: U:m*r, S:r*r, V:n*r.
    """
    mat = sp.Matrix(mat)
    if mat.rows == 0 or mat.cols == 0:
        raise ValueError('Expected a nonempty matrix.')
    if mat.is_zero_matrix:
        return sp.zeros(*mat.shape)
    rank = mat.rank()
    if rank == 1:
        return simplify_matrix(mat / frob(mat))
    if mat.shape == (2, 2):
        a, b, c, d = list(mat)
        determinant = sp.simplify(mat.det())
        if determinant.is_positive:
            candidate = sp.Matrix([[a+d, b-c], [c-b, a+d]])
        elif determinant.is_negative:
            candidate = sp.Matrix([[a-d, b+c], [b+c, d-a]])
        else:
            raise ValueError('The 2x2 certificate requires a known determinant sign.')
        # Candidate has two equal singular values: normalize a single row.
        return simplify_matrix(candidate / sp.sqrt(sum(x*x for x in candidate.row(0))))
    u, singular, v = mat.singular_value_decomposition()
    assert u.cols == rank and singular.shape == (rank, rank) and v.cols == rank
    return simplify_matrix(u * v.T)


def ema_step(state, grad, mu=q(1, 2)):
    return simplify_matrix(mu*state + (1-mu)*grad)


def adam_direction(grad, first, second, count, beta1=q(1, 2),
                   beta2=q(3, 4), epsilon=q(1, 10)):
    """One state update; count is the resulting actual state-update count."""
    if count < 1:
        raise ValueError('Adam count must be positive.')
    first = simplify_matrix(beta1*first + (1-beta1)*grad)
    second = simplify_matrix(beta2*second + (1-beta2)*grad.multiply_elementwise(grad))
    first_hat = first / (1-beta1**count)
    second_hat = second / (1-beta2**count)
    direction = sp.Matrix(grad.rows, grad.cols,
                          [sp.simplify(a/(sp.sqrt(b)+epsilon))
                           for a, b in zip(first_hat, second_hat)])
    return direction, first, second, first_hat, second_hat


def power_transform(mat, steps=5, polynomial='cubic', precision=None, epsilon=0):
    """Same-shape finite transform, one Frobenius normalization, smaller Gram.

    epsilon=0 is the teaching recipe. At precision=None calculations are
    exact; use at most two exact small steps to avoid expression explosion.
    """
    if not isinstance(steps, int) or steps < 0:
        raise ValueError('steps must be a nonnegative integer.')
    if polynomial not in ('cubic', 'quintic'):
        raise ValueError('Unknown polynomial.')
    mat = sp.Matrix(mat)
    if mat.is_zero_matrix:
        return sp.zeros(*mat.shape)
    denom = frob(mat) + sp.sympify(epsilon)
    x = mat / denom
    if precision is not None:
        x = x.evalf(precision)
    a, b, c = ((q(3, 2), -q(1, 2), 0) if polynomial == 'cubic' else QUINTIC)
    for _ in range(steps):
        if x.rows <= x.cols:
            gram = x*x.T
            x = a*x + (b*gram + c*gram*gram)*x
        else:
            gram = x.T*x
            x = a*x + x*(b*gram + c*gram*gram)
        x = x.evalf(precision) if precision is not None else simplify_matrix(x)
    return x


def measured_rms_direction(mat, target=1):
    if mat.is_zero_matrix:
        return sp.zeros(*mat.shape)
    return simplify_matrix(target*mat/rms(mat))


def headwise_polar(mat):
    return sp.Matrix.vstack(*(supported_polar(mat.row(i)) for i in range(mat.rows)))


def error_pair(iterate, exact, precision=PRECISION):
    exact = exact.evalf(precision)
    error = frob(iterate-exact).evalf(precision)
    residual = frob(iterate.T*iterate-exact.T*exact).evalf(precision)
    return error, residual


@dataclass(eq=False)
class Parameter:
    """One identity-owned parameter with optional absent-gradient support."""
    name: str
    value: sp.MatrixBase
    method: str
    decay: sp.Expr = q(1, 5)
    count: int = 0
    momentum: sp.MatrixBase = field(init=False)
    first: sp.MatrixBase = field(init=False)
    second: sp.MatrixBase = field(init=False)

    def __post_init__(self):
        self.value = sp.Matrix(self.value)
        self.momentum = sp.zeros(*self.value.shape)
        self.first = sp.zeros(*self.value.shape)
        self.second = sp.zeros(*self.value.shape)
        if self.method not in ('matrix', 'adamw'):
            raise ValueError('Unknown parameter method.')

    def step(self, grad, eta=q(1, 10), exact=False, ns_steps=5):
        if grad is None:
            return  # Skip states, count, decay and update together.
        grad = sp.Matrix(grad)
        if grad.shape != self.value.shape:
            raise ValueError('Gradient shape differs from parameter.')
        self.count += 1
        if self.method == 'matrix':
            self.momentum = ema_step(self.momentum, grad)
            direction = (supported_polar(self.momentum) if exact else
                         power_transform(self.momentum, ns_steps, precision=PRECISION))
            direction *= sp.sqrt(max(grad.shape))
        else:
            direction, self.first, self.second, _, _ = adam_direction(
                grad, self.first, self.second, self.count)
        self.value = (1-eta*self.decay)*self.value - eta*direction
        if exact or self.method == 'adamw':
            self.value = simplify_matrix(self.value)


def update_tied_uses(uses, eta=q(1, 10), exact=False, ns_steps=5):
    """Sum gradients for aliases of a Parameter, then perform one state update.

    Each use is (Parameter, gradient-or-None). All absent aliases skip. Distinct
    Parameter objects may not share the same value storage; reject that policy.
    """
    owners, combined, storage = {}, {}, {}
    for parameter, grad in uses:
        identity, value_id = id(parameter), id(parameter.value)
        if value_id in storage and storage[value_id] != identity:
            raise ValueError('Shared storage has contradictory parameter ownership.')
        storage[value_id] = identity
        owners[identity] = parameter
        if grad is not None:
            grad = sp.Matrix(grad)
            if grad.shape != parameter.value.shape:
                raise ValueError('Tied use gradient shape mismatch.')
            combined[identity] = combined.get(identity, sp.zeros(*grad.shape)) + grad
    for identity, parameter in owners.items():
        parameter.step(combined.get(identity), eta, exact=exact, ns_steps=ns_steps)


def run_checks():
    """Independent fixture identities and counterexamples, counted by property."""
    checks = []

    def check(name, condition):
        assert bool(condition), name
        checks.append(name)

    losses, _, gradients, mean_grad = symbolic_gradients()
    subs = dict.fromkeys(sp.symbols('w11 w12 w21 w22', real=True), 0)
    check('A: initial target loss', equal(losses[0].subs(subs), sp.log(4)))
    check('A: first symbolic gradient', equal(gradients[0], G1))
    check('A: second symbolic gradient', equal(gradients[1], G2))
    check('A: independently differentiated mean', equal(mean_grad, GBAR))
    check('A: single-example and batch ranks', (G1.rank(), G2.rank(), GBAR.rank()) == (1, 1, 2))
    check('A: batch determinant', equal(GBAR.det(), -q(1, 12)))
    b1 = ema_step(sp.zeros(2), G1)
    b2 = ema_step(b1, G2)
    check('B: raw EMA replay', equal(b2, sp.Matrix([[q(1, 24), -q(31, 96)], [-q(1, 8), -q(1, 32)]])))
    check('B: startup correction weights', equal(b2/(1-q(1, 2)**2), (G1+2*G2)/3))
    direction, first, second, fh, sh = adam_direction(GBAR, sp.zeros(2), sp.zeros(2), 1)
    check('B: first raw moment', equal(first, GBAR/2))
    check('B: second raw moment', equal(second, GBAR.multiply_elementwise(GBAR)/4))
    check('B: both Adam correction factors', equal(fh, GBAR) and equal(sh, GBAR.multiply_elementwise(GBAR)))
    check('B: epsilon-outside-root direction', equal(direction, DA))
    inside = sp.Matrix(2, 2, [g/sp.sqrt(v+q(1,10)) for g,v in zip(fh, sh)])
    check('B: inside-root mutation detected', not equal(inside, DA))
    constant = sp.ones(2)
    _, _, _, _, constant_second = adam_direction(constant, sp.zeros(2), sp.zeros(2), 1)
    check('B: second moment is not variance', equal(constant_second, constant))
    w, pre = sp.ones(2, 1), sp.diag(1, 4)
    coupled = w-q(1,10)*pre*(q(1,5)*w)
    decoupled = (1-q(1,10)*q(1,5))*w
    check('B: fixed-P decay counterexample', equal(coupled, sp.Matrix([q(49,50),q(23,25)])) and equal(decoupled, sp.Matrix([q(49,50),q(49,50)])))
    for label, mat in [('token',G1),('batch',GBAR),('positive',M),('wide',A),('tall',A.T),('deficient',A0),('deficient tall',A0.T),('zero',sp.zeros(2,3))]:
        polar = supported_polar(mat)
        magnitude = simplify_matrix(polar.T*mat)
        check('C: '+label+' reconstruction and symmetric magnitude', equal(polar*magnitude,mat) and equal(magnitude,magnitude.T))
        check('C: '+label+' positive semidefinite magnitude', magnitude.is_positive_semidefinite is True)
        row, col = simplify_matrix(polar*polar.T), simplify_matrix(polar.T*polar)
        check('C: '+label+' support projectors', equal(row*row,row) and equal(col*col,col) and row.rank()==mat.rank() and col.rank()==mat.rank())
        check('C: '+label+' transpose consistency', equal(supported_polar(mat.T),polar.T))
    check('C: exact token certificate', equal(supported_polar(G1),Q1))
    check('C: exact batch certificate', equal(supported_polar(GBAR),QBAR))
    h = simplify_matrix(QBAR.T*GBAR)
    check('C: batch positive magnitude', equal(QBAR.T*QBAR,sp.eye(2)) and h[0,0]>0 and h.det()>0)
    check('C: deficient Gram is not identity', not equal(supported_polar(A0)*supported_polar(A0).T,sp.eye(2)))
    check('C: positive scale equivariance', equal(supported_polar(10*GBAR),QBAR))
    s = sp.symbols('s', real=True)
    phi = (3*s-s**3)/2
    check('D: progress factorization', equal(phi-s,s*(1-s*s)/2))
    check('D: no-overshoot factorization', equal(1-phi,(1-s)**2*(s+2)/2))
    e = sp.symbols('e', real=True)
    residual = sp.expand(1-phi**2).subs(s**6,(1-e)**3).subs(s**4,(1-e)**2).subs(s**2,1-e)
    check('D: singular residual recurrence', equal(residual,e**2*(3+e)/4))
    s1, s2 = phi.subs(s,q(1,2)), phi.subs(s,phi.subs(s,q(1,2)))
    check('D: two rational scalar iterates', equal(s1,q(11,16)) and equal(s2,q(7117,8192)))
    check('D: quintic does not fix one', sum(QUINTIC)==q(701,1000))
    for polynomial in ('cubic','quintic'):
        check('D: '+polynomial+' zero branch', power_transform(sp.zeros(3,2),5,polynomial).shape==(3,2) and equal(power_transform(sp.zeros(3,2),5,polynomial),sp.zeros(3,2)))
        for count in (0,1,2):
            exact = power_transform(A,count,polynomial)
            numeric = power_transform(A,count,polynomial,precision=PRECISION)
            check(f'D: {polynomial} exact/numeric agreement at {count}', frob(exact.evalf(PRECISION)-numeric).evalf(PRECISION)<AGREEMENT_TOLERANCE)
        transposed = power_transform(A.T,5,polynomial,precision=PRECISION)
        check('D: '+polynomial+' finite transpose consistency', frob(transposed-power_transform(A,5,polynomial,precision=PRECISION).T)<AGREEMENT_TOLERANCE)
    check('D: token cubic already supported', equal(power_transform(G1,2),Q1))
    for label,mat in [('wide',A),('tall',A.T),('deficient',A0),('deficient tall',A0.T)]:
        polar=supported_polar(mat); rank=mat.rank(); m,n=mat.shape
        check('E: '+label+' exact rank RMS', equal(rms(polar),sp.sqrt(q(rank,m*n))))
        check('E: '+label+' shape-scaled RMS', equal(rms(sp.sqrt(max(m,n))*polar),sp.sqrt(q(rank,min(m,n)))))
    finite=power_transform(A,1)
    check('E: measured RMS is a distinct policy', equal(rms(measured_rms_direction(finite)),1) and not equal(rms(sp.sqrt(3)*finite),1))
    check('E: measured RMS zero safeguard', equal(measured_rms_direction(sp.zeros(2)),sp.zeros(2)))
    whole,heads = supported_polar(MH), headwise_polar(MH)
    check('E: whole factor', equal(whole,sp.Matrix([[4,-1],[1,4]])/sp.sqrt(17)))
    check('E: head Gram', equal(heads*heads.T,sp.Matrix([[1,1/sp.sqrt(2)],[1/sp.sqrt(2),1]])))
    check('E: equal RMS different geometry', equal(rms(whole),rms(heads)) and not equal(whole,heads))
    scaled=sp.diag(10,1)*MH
    check('E: head positive-scale invariance versus whole coupling', equal(headwise_polar(scaled),heads) and not equal(supported_polar(scaled),whole))
    check('E: functional moment counterexample', (sp.Matrix([[1,0]])*sp.diag(4,1)*sp.Matrix([[1],[0]]))[0] != (sp.Matrix([[0,1]])*sp.diag(4,1)*sp.Matrix([[0],[1]]))[0])
    expected=-sp.Matrix([[13,-25],[-25,-13]])/(10*sp.sqrt(397))
    projection=Parameter('projection',W0,'matrix')
    projection.step(GBAR,exact=True)
    check('F: complete first exact update', equal(projection.value,expected))
    nonzero=Parameter('nonzero',sp.ones(2),'matrix')
    nonzero.step(GBAR,exact=True)
    wanted=q(49,50)*sp.ones(2)-q(1,10)*sp.sqrt(2)*QBAR
    check('F: decay kept out of polar input', equal(nonzero.value,wanted))
    for shape in ((2,2),(2,3)):
        parameter=Parameter('decay only',sp.ones(*shape),'matrix')
        parameter.step(sp.zeros(*shape),exact=True)
        check('F: decay unscaled for shape '+str(shape),equal(parameter.value,q(49,50)*sp.ones(*shape)))
    missing=Parameter('optional',sp.ones(2),'adamw')
    missing.step(None)
    check('F: absent skips state count and decay',missing.count==0 and equal(missing.value,sp.ones(2)))
    missing.step(GBAR)
    check('F: actual-state Adam count after skip',missing.count==1 and equal(missing.first,GBAR/2) and equal(missing.value,q(49,50)*sp.ones(2)-q(1,10)*DA))
    missing.step(sp.zeros(2))
    check('F: explicit zero advances state',missing.count==2 and equal(missing.first,GBAR/4))
    tied=Parameter('tied embedding/readout',sp.zeros(2),'adamw')
    update_tied_uses([(tied,G1),(tied,G2)])
    expected_dir=adam_direction(G1+G2,sp.zeros(2),sp.zeros(2),1)[0]
    check('F: tied aliases sum and update once',tied.count==1 and equal(tied.first,(G1+G2)/2) and equal(tied.value,-q(1,10)*expected_dir))
    adam0=Parameter('adam ns0',W0,'adamw'); adam10=Parameter('adam ns10',W0,'adamw')
    update_tied_uses([(adam0,GBAR)],ns_steps=0); update_tied_uses([(adam10,GBAR)],ns_steps=10)
    check('F: Adam count independent of NS loop',adam0.count==adam10.count==1 and equal(adam0.value,adam10.value))
    check('F: uphill momentum counterexample',sp.trace(sp.eye(2).T*(-supported_polar(-sp.eye(2))))==2)
    return checks


def comparison_rows():
    """Full requested finite sweep, for full rank, rectangular and deficient cases."""
    rows=[]
    for name,mat in [('batch',GBAR),('wide',A),('tall',A.T),('deficient',A0),('token',G1),('zero',sp.zeros(2,3))]:
        exact=supported_polar(mat)
        for polynomial in ('cubic','quintic'):
            for count in COUNTS:
                iterate=power_transform(mat,count,polynomial,precision=PRECISION)
                error,residual=error_pair(iterate,exact)
                rows.append((name,polynomial,count,error,residual))
    return rows


def direction_comparison_rows():
    """Task F's complete first-step comparison, including both finite polynomials.

    Raw SGD and Adam use no decay, as in the chapter's initial comparison.
    Matrix variants use the declared mu=1/2 momentum input, sqrt(2) scale,
    eta=1/10 and lambda=1/5. Old W0 is zero, so every decay change is zero.
    Finite directions use five steps with identical once-only normalization.
    """
    matrix_input = ema_step(sp.zeros(2), GBAR)
    adam = adam_direction(GBAR, sp.zeros(2), sp.zeros(2), 1)[0]
    variants = [
        ('SGD', GBAR, sp.Integer(1), None, sp.Integer(0)),
        ('Adam', adam, sp.Integer(1), None, sp.Integer(0)),
        ('exact polar', supported_polar(matrix_input), sp.sqrt(2), None, q(1, 5)),
    ]
    for polynomial in ('cubic', 'quintic'):
        raw = power_transform(matrix_input, 5, polynomial, precision=PRECISION)
        variants.append((polynomial, raw, sp.sqrt(2), 5, q(1, 5)))
    rows = []
    for method, raw, scale, iterations, decay in variants:
        direction = scale*raw
        loss_change = -q(1, 10)*direction
        decay_change = -q(1, 10)*decay*W0
        rows.append({
            'method': method, 'iterations': iterations, 'scale': scale,
            'raw_direction': raw, 'direction': direction,
            'loss_change': loss_change, 'decay_change': decay_change,
            'parameter': W0+loss_change+decay_change,
        })
    return rows


def main():
    print('Python',sys.version.split()[0],'SymPy',sp.__version__)
    print('Arithmetic:',PRECISION,'digits; agreement tolerance',AGREEMENT_TOLERANCE)
    checks=run_checks()
    print(len(checks),'distinct mathematical/behavioral checks passed.')
    print('fixture polynomial L factor_error column_support_Gram_residual')
    for name,polynomial,count,error,residual in comparison_rows():
        print(name,polynomial,count,str(error.evalf(9)),str(residual.evalf(9)))
    print('Task F: first equal-weight minibatch at W0=0; no clipping;'
          ' whole 2x2 hidden projection; readout and upstream features frozen.')
    print('SGD uses Gbar with no momentum. Adam uses beta1=1/2, beta2=3/4,'
          ' zero states, epsilon_A=1/10 outside the root.')
    print('Matrix input: B1=Gbar/2, mu=1/2, zero state, no Nesterov mixture;'
          ' Frobenius normalization once, epsilon_N=0, same-shape zero branch.')
    print('Finite transforms: five cubic/quintic steps, restored column-input'
          ' orientation, 80 digits; exact rows retain symbolic arithmetic.')
    print('eta=1/10; SGD/Adam decay=0; matrix decay=1/5 on old W;'
          ' matrix direction scale=sqrt(2). Finite entries below rounded to 9 digits.')
    print('method | L | scale | raw_direction | nominal_direction |'
          ' loss_change | decay_change | W1')
    directions = direction_comparison_rows()
    for row in directions:
        finite = row['iterations'] is not None
        matrices = [row[key] for key in ('raw_direction', 'direction',
                                         'loss_change', 'decay_change', 'parameter')]
        if finite:
            matrices = [mat.evalf(9) for mat in matrices]
        print(' | '.join([row['method'], str(row['iterations']), str(row['scale'])]
                         + [str(mat) for mat in matrices]))
    by_method = {row['method']: row for row in directions}
    exact_update = by_method['exact polar']['parameter']
    finite_update = by_method['cubic']['parameter']
    print('Cubic5 update error:',frob(finite_update-exact_update.evalf(PRECISION)).evalf(12))


if __name__=='__main__':
    main()
