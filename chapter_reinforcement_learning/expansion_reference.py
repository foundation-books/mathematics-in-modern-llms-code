"""Exact CPU reference for the expanded homework; constructed mathematics only."""
from __future__ import annotations
import json
import platform
import sympy as sp
R = sp.Rational
CHECKS = []
FAULTS = []
def check(ok, label):
    if not bool(ok): raise AssertionError(label)
    CHECKS.append(label)
def eq(a, b, label):
    if isinstance(a, sp.MatrixBase) or isinstance(b, sp.MatrixBase):
        d = sp.Matrix(a)-sp.Matrix(b)
        check(all(sp.simplify(v)==0 for v in d), label)
    else: check(sp.simplify(a-b)==0, label)
def fault(ok, label):
    check(ok, label); FAULTS.append(label)
def rejected(fn, label):
    try: fn()
    except ValueError: fault(True, label)
    else: raise AssertionError(label)
def report(data):
    return dict(python=platform.python_version(), sympy=sp.__version__,
                checks=len(CHECKS), fault_checks=FAULTS, **data)

from itertools import product

def local_score(ell,A=R(1,4),lam=R(1,10),a=-R(1,5),b=R(1,5)):
    if not a<b or lam<0:raise ValueError('local domain')
    return A*sp.exp(min(max(ell,a),b))-lam*ell**2

def local_coefficients(ell,A=R(1,4),lam=R(1,10),a=-R(1,5),b=R(1,5)):
    reward=A*sp.exp(ell) if a<=ell<=b else sp.S.Zero
    return reward,-2*lam*ell,reward-2*lam*ell

def admitted(q,a):
    if len(q)!=len(a) or sum(q)!=1 or any(x<0 for x in q) or any(x<0 or x>1 for x in a):raise ValueError('admission law')
    mass=sum(qq*aa for qq,aa in zip(q,a))
    if mass<=0:raise ValueError('no admissions')
    return mass,[qq*aa/mass for qq,aa in zip(q,a)]

def correction(p,q,a):
    if len(p)!=len(q) or len(q)!=len(a) or sum(p)!=1 or sum(q)!=1 or any(x<0 for x in [*p,*q]) or any(x<0 or x>1 for x in a):raise ValueError('joint law domain')
    if any(pp>0 and qq*aa==0 for pp,qq,aa in zip(p,q,a)):raise ValueError('target outcome unsupported')
    return [pp/(qq*aa) if qq*aa else sp.S.Zero for pp,qq,aa in zip(p,q,a)]


def weight_and_group_checks():
    """Exact elective B/D/G extensions; no production epsilon is inferred."""
    ratio = sp.symbols('positive_ratio', positive=True)
    eps = R(1, 5)
    clipped = sp.Min(sp.Max(ratio, 1-eps), 1+eps)
    ratios = (R(1, 2), sp.S.One, sp.Integer(2), sp.Integer(4))
    # Differentiate PPO's forward surrogate, then convert d/dr to a log-score coefficient.
    expected = {1: (R(1, 2), 1, 0, 0), -1: (0, -1, -2, -4)}
    coefficient_table = []
    for advantage in (1, -1):
        objective = sp.Min(advantage*ratio, advantage*clipped)
        coefficients = tuple(sp.simplify((ratio*sp.diff(objective, ratio)).subs(ratio, r))
                             for r in ratios)
        check(coefficients == expected[advantage], 'D PPO coefficient table sign '+str(advantage))
        detached = tuple(advantage*min(r, 3) for r in ratios)
        check(detached == tuple(advantage*r for r in (R(1, 2), 1, 2, 3)),
              'D detached coefficient table sign '+str(advantage))
        coefficient_table.append(dict(advantage=advantage, detached=list(map(str, detached)),
                                      ppo=list(map(str, coefficients))))
    eq(sp.diff(sp.S.Zero*ratio, ratio), 0, 'D zero advantage')
    eq(min(sp.Integer(2), R(6, 5)), R(6, 5), 'D shared upper threshold remains active')
    eq((ratio*sp.diff(sp.Min(ratio, clipped), ratio)).subs(ratio, 2), 0,
       'D PPO shared-threshold zero')
    logit, frozen, advantage = sp.symbols('binary_logit frozen_weight advantage', real=True)
    current = 1/(1+sp.exp(-logit)); generator = R(1, 8)
    score = sp.diff(sp.log(current), logit)
    prescribed = sp.diff(advantage*frozen*sp.log(current), logit)
    eq(prescribed, advantage*frozen*score, 'D independent frozen weight derivative')
    eq(score.subs(logit, 0), R(1, 2), 'D normalized binary score at ratio four')
    eq((current/generator).subs(logit, 0), 4, 'D binary ratio four')
    eq(prescribed.subs({advantage: 1, frozen: 3, logit: 0}), R(3, 2),
       'D normalized detached ascent derivative')
    below_cap = advantage*(current/generator)*sp.log(current)
    product_rule = advantage*(current/generator)*(1+sp.log(current))*score
    eq(sp.diff(below_cap, logit), product_rule, 'D undetached product-rule derivative')
    # At ratio one, recomputing the ratio differentiates another forward expression.
    evaluation = -sp.log(7)
    correct = prescribed.subs({advantage: 1, frozen: 1, logit: evaluation})
    wrong = sp.diff(below_cap, logit).subs({advantage: 1, logit: evaluation})
    check(sp.simplify(correct-wrong) != 0, 'D ratio recomputation changes derivative')
    frozen_function = sp.log(current)  # frozen coefficient=1 at this point
    errors = []
    for step in (R(1, 100), R(1, 10000), R(1, 1000000)):
        difference = (frozen_function.subs(logit, evaluation+step).evalf(60)
                      - frozen_function.subs(logit, evaluation-step).evalf(60))/(2*step)
        error = abs(difference-correct.evalf(60)); errors.append(error)
    check(errors[2] < errors[1] < errors[0] < R(1, 100000),
          'D frozen-coefficient central differences converge')

    groups = ((sp.S.Zero, sp.S.One), (R(2, 5), R(3, 5)))
    def center(batch):
        return tuple(tuple(x-sum(group)/sp.Integer(len(group)) for x in group) for group in batch)
    def pooled(batch):
        centered = center(batch)
        count = sum(map(len, centered))
        variance = sum(x*x for group in centered for x in group)/count
        normalized = tuple(tuple(x/sp.sqrt(variance) if variance else sp.S.Zero for x in group)
                           for group in centered)
        return centered, variance, normalized
    centered, variance, normalized = pooled(groups)
    check(centered == ((-R(1, 2), R(1, 2)), (-R(1, 10), R(1, 10))), 'B centered fixture')
    check(all(sum(group) == 0 for group in centered), 'B each centered sum zero')
    eq(variance, R(13, 100), 'B pooled centered population variance')
    check(normalized == ((-5/sp.sqrt(13), 5/sp.sqrt(13)),
                         (-1/sp.sqrt(13), 1/sp.sqrt(13))), 'B pooled standardized coefficients')
    eq(normalized[0][1]/normalized[1][1], 5, 'B coefficient magnitude ratio five')
    for group in centered:
        group_scale = sp.sqrt(sum(x*x for x in group)/len(group))
        check(tuple(x/group_scale for x in group) == (-1, 1), 'B separate group scale')
    # General unequal-size identity from independently centered symbolic rewards.
    unequal = (sp.symbols('group_a0:2', real=True), sp.symbols('group_b0:3', real=True))
    centered_unequal, pooled_variance, _ = pooled(unequal)
    group_variances = [sum(x*x for x in group)/len(group) for group in centered_unequal]
    eq(pooled_variance, sum(R(len(group), 5)*var for group, var in zip(unequal, group_variances)),
       'B unequal-size N_g/B variance identity')
    numeric_unequal = ((sp.S.Zero, sp.S.One), (sp.S.Zero, sp.S.One, sp.Integer(2)))
    eq(pooled(numeric_unequal)[1], R(1, 2), 'B unequal-size exact pooled variance')
    check(R(1, 2) != (R(1, 4)+R(2, 3))/2, 'B equal group weights define another variance')
    translated = (tuple(x+7 for x in groups[0]), groups[1])
    check(pooled(translated) == pooled(groups), 'B within-group translation invariance')
    scaled = tuple(tuple(3*x for x in group) for group in groups)
    check(pooled(scaled)[2] == normalized, 'B common positive reward scale invariance')
    raw = tuple(x for group in translated for x in group)
    raw_mean = sum(raw)/len(raw)
    check(sum((x-raw_mean)**2 for x in raw)/len(raw) != variance, 'B raw reward variance differs')
    check(pooled(((sp.Integer(2),)*2, (sp.Integer(7),)*3))[2] == ((0, 0), (0, 0, 0)),
          'B zero pooled variance teaching rule')
    check(sum(centered[0][1:]) != 0, 'B individual filtering destroys a centered group sum')

    token_groups = (((1,), (-1, -1, -1)), ((1,), (1,)))
    count = sum(map(len, token_groups))
    group_mean = sum(R(len(group), sum(map(len, group)))*sum(map(sum, group))
                     for group in token_groups)/count
    response_mean = sum(R(sum(response), len(response)) for group in token_groups
                        for response in group)/count
    total = sum(sum(response) for group in token_groups for response in group)
    tokens = sum(len(response) for group in token_groups for response in group)
    eq(group_mean, R(1, 4), 'G group token reduction')
    eq(response_mean, R(1, 2), 'G response mean reduction')
    eq(R(total, tokens), 0, 'G global token reduction')
    for group in token_groups:
        token_count = sum(map(len, group))
        eq(token_count*R(len(group), token_count), len(group), 'G base group weights sum to N_g')
    return dict(coefficients=coefficient_table, pooled_variance=str(variance),
                group_reductions=list(map(str, (group_mean, response_mean, R(total, tokens)))),
                frozen_difference_errors=list(map(str, errors)))

def run_checks():
    # Recall Chapter 18's canonical tilt.
    p0=[R(1,2),R(1,3),R(1,6)];r=[sp.log(2),sp.S.Zero,-sp.log(2)]
    Z=sum(pp*sp.exp(rr) for pp,rr in zip(p0,r));tilt=[pp*sp.exp(rr)/Z for pp,rr in zip(p0,r)]
    eq(Z,R(17,12),'canonical Chapter 18 partition');check(tilt==[R(12,17),R(4,17),R(1,17)],'tilted response law')
    for pp,ref,reward in zip(tilt,p0,r):eq(reward-sp.log(pp/ref),sp.log(Z),'centered residual retains partition')
    fault(Z!=1,'centering does not set log partition to zero')
    eq(sum(tilt), 1, 'B1 recalled tilt is normalized')

    # B1: exact teaching inputs for the effort-conditioned capped charge.
    ell = sp.symbols('length', nonnegative=True)
    k0, normalizer, cap = sp.S.One, sp.Integer(4), sp.S.One
    tau = sp.symbols('tau_eff', positive=True)
    effort_table = []
    for increment, expected_cap, expected_charges in (
        (0, 4, (0, -R(1, 2), -1, -1)),
        (tau * sp.log(2), 8, (0, -R(1, 4), -R(1, 2), -1)),
    ):
        k = k0 * sp.exp(-increment / tau)
        cap_at = sp.solve(k * ell / normalizer - cap, ell)[0]
        eq(cap_at, expected_cap, 'B1 saturation length')
        charges = tuple(-min(cap, k * l / normalizer) for l in (0, 2, 4, 8))
        check(charges == expected_charges, 'B1 charge table')
        # Check the one-sided slopes by evaluating the charge on either side.
        step = sp.symbols('step', positive=True)
        left = sp.limit((-k * (cap_at - step) / normalizer + cap) / (-step), step, 0, dir='+')
        right = sp.limit((-cap + cap) / step, step, 0, dir='+')
        eq(left, -k / normalizer, 'B1 below-cap slope')
        eq(right, 0, 'B1 saturated slope')
        effort_table.append(dict(k=str(k), cap_length=str(cap_at),
                                 below_cap_slope=str(left), above_cap_slope=str(right),
                                 charges=list(map(str, charges))))
    effort = sp.symbols('effort_above_minimum', nonnegative=True)
    coeff = k0 * sp.exp(-effort / tau)
    eq(coeff.subs(effort, effort + tau) / coeff, sp.exp(-1),
       'B1 multiplier for one effort scale')
    B=[R(1,4)]*4;P=[R(1,4),R(1,2),R(1,8),R(1,8)]
    eq(sum(P),1,'normalized current leaf law');eq(sum(B),1,'normalized behavior leaf law')
    l1=sp.log(R(3,2));l2=sp.log(R(2,3))
    eq(l1+l2,0,'sample path cancellation');eq(l1*l1+l2*l2,2*l1*l1,'local squares survive cancellation')
    kl=sum(pp*sp.log(pp/bb) for pp,bb in zip(P,B));eq(kl,sp.log(2)/4,'positive full current-to-behavior KL')
    fault(kl>0,'sample path equality does not identify full law')
    ell=sp.symbols('ell',real=True);inner=sp.exp(ell)/4-ell**2/10
    table=[]
    for point in [-R(3,10),-R(1,10),R(1,10),R(3,10)]:
        reward,restore,total=local_coefficients(point)
        oracle=sp.diff(inner,ell).subs(ell,point) if abs(point)<R(1,5) else -point/5
        eq(total,oracle,'local derivative independent region formula');table.append([str(point),str(reward),str(restore),str(total)])
    a=-R(1,5);b=R(1,5)
    for endpoint,side in [(a,'+'),(b,'-')]:
        f=sp.exp(ell)/4-ell**2/10;f0=f.subs(ell,endpoint)
        one_sided=sp.limit((f-f0)/(ell-endpoint),ell,endpoint,dir=side)
        eq(one_sided,local_coefficients(endpoint)[2],'selected interior one-sided boundary')
        eq(local_coefficients(endpoint,A=0)[2],sp.diff(-ell**2/10,ell).subs(ell,endpoint),'zero-advantage quadratic smooth at a reward boundary')
    eq(local_coefficients(R(3,10))[2],-R(3,50),'exterior restoration despite positive advantage')
    fault(local_coefficients(R(3,10))[2]!=0,'whole-score eligibility mask wrongly removes restoration')
    fault(local_score(R(3,10))!=-R(9,1000),'indicator-exponential changes held forward value')
    u=sp.symbols('u',real=True);pu=1/(1+sp.exp(-u));lr=sp.log(2*pu)
    tail=sp.exp(b)/4-lr**2/10;du=sp.simplify(sp.diff(tail,u).subs(u,sp.log(3)))
    eq(du,-sp.log(R(3,2))/20,'normalized Bernoulli derivative')
    unew=sp.log(3)+du/10;pnew=1/(1+sp.exp(-unew));eq(unew,sp.log(3)-sp.log(R(3,2))/200,'exact ascent logit')
    check(float(pnew.evalf(30))<0.75,'positive-advantage selected action can decrease')
    leaked=sp.log(pu/pu);fault(sp.diff(leaked,u)==0 and du!=0,'differentiating through behavior erases score')
    token_fragments = ((1, 1), (-1,))
    lengths = tuple(map(len, token_fragments))
    means = tuple(R(sum(fragment), len(fragment)) for fragment in token_fragments)
    eq(sum(n*m for n, m in zip(lengths, means)) / sum(lengths), R(1, 3),
       'F length-weighted mean of (1,1)|(-1)')
    eq(sum(means) / len(means), 0, 'F equal mean of the two fragments')
    whole=(u-u+2*u)/3;fragments=((u-u)/2+2*u)/2
    eq(whole.subs(u,0),fragments.subs(u,0),'equal values at zero')
    eq(sp.diff(whole,u),R(2,3),'logical token-mean derivative');eq(sp.diff(fragments,u),1,'wrong fragment-mean derivative')
    fault(sp.diff(whole-fragments,u)!=0,'equal-value different-gradient repacking')
    p=[R(1,3)]*3;q=[R(1,15),R(1,3),R(3,5)];a=[1,R(1,2),R(1,4)];f=[1,0,-1]
    mass,qa=admitted(q,a);eq(mass,R(23,60),'task admission probability');check(qa==[R(4,23),R(10,23),R(9,23)],'admitted task law')
    old=[pp/qq for pp,qq in zip(p,q)];wrong=sum(qq*ww*ff for qq,ww,ff in zip(qa,old,f));eq(wrong,R(15,23),'task-only admitted bias')
    attempt_w=correction(p,q,a);admitted_w=[mass*ww for ww in attempt_w]
    check(admitted_w==[R(23,12),R(23,30),R(23,27)],'admitted weights with mass factor')
    eq(sum(qq*ww*ff for qq,ww,ff in zip(qa,admitted_w,f)),0,'admitted-denominator correction')
    eq(sum(qq*aa*ww*ff for qq,aa,ww,ff in zip(q,a,attempt_w,f)),0,'attempt-denominator correction')
    fault(sum(qq*ww for qq,ww in zip(qa,attempt_w))!=1,'per-attempt weight averaged over admissions is wrong on constant integrand')
    # Use positive integrand to detect missing Abar even when the desired signed mean is zero.
    eq(sum(qq*ww for qq,ww in zip(qa,attempt_w)),1/mass,'wrong admitted normalization on constant integrand')
    m,selected=admitted([R(1,2)]*2,[1,R(1,4)])
    eq(m,R(5,8),'outcome admission');eq(selected[0]-selected[1],R(3,5),'within-task selection bias')
    eq(sum(bb*aa*ww*ff for bb,aa,ww,ff in zip([R(1,2)]*2,[1,R(1,4)],correction([R(1,2)]*2,[R(1,2)]*2,[1,R(1,4)]),[1,-1])),0,'outcome-specific correction')
    rejected(lambda:correction([R(1,2)]*2,[R(1,2)]*2,[1,0]),'zero inclusion outcome unidentifiable')
    rejected(lambda:admitted([2,-1,0],[1,1,1]),'negative acquisition mass rejected')
    rejected(lambda:correction([R(1,2)]*2,[R(1,2)]*2,[1,R(5,4)]),'inclusion above one rejected')
    # Joint task/current-trajectory/inclusion law on six specified events.
    current=[[R(3,4),R(1,4)],[R(1,3),R(2,3)],[R(1,2)]*2];target_mean=joint=sp.S.Zero
    for i,j in product(range(3),range(2)):
        reward=[1,-1][j];inc=[1,R(1,4)][j];beh=R(1,2)
        target_mean+=p[i]*current[i][j]*reward
        joint+=q[i]*beh*inc*(p[i]*current[i][j]/(q[i]*beh*inc))*reward
    eq(joint,target_mean,'joint task trajectory inclusion expectation');eq(joint,R(1,18),'joint exact scalar')
    fault(R(1,2)!=R(1,2)**2,'correlated group admission not product of marginals')
    # Ratio denominator must be the transformed generator, not the raw model.
    fault(R(3,4)/sp.S.One!=R(3,4)/R(1,2),'raw denominator after behavior support filtering')
    weighting = weight_and_group_checks()
    return report(dict(mistral_weighting=weighting, partition=str(Z),tilt=[str(x) for x in tilt],
                       effort_length_charges=effort_table, sequence_KL=str(kl),
                       coefficient_table=table,new_logit=str(unew),new_probability=str(pnew),
                       admission=str(mass),admitted=[str(x) for x in qa],
                       wrong_mean=str(wrong),joint_mean=str(joint)))

if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
