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

def factors(scores,threshold):
    threshold=sp.sympify(threshold)
    if not scores or threshold.is_finite is not True or threshold.is_positive is not True:raise ValueError('threshold/statistics')
    vals=[sp.sympify(s) for s in scores]
    if any(s.is_finite is not True or s.is_real is not True for s in vals):raise ValueError('nonfinite statistic')
    return [threshold/s if s>threshold else sp.S.One for s in vals]

def run_checks():
    z=sp.Matrix([R(2,3),R(5,6)]); qs=[sp.Matrix([[30,0]]),sp.Matrix([[12,0]])];ks=[sp.Matrix([[30,0]]),sp.Matrix([[15,0]])]
    scores=[(q*z)[0]*(k*z)[0] for q,k in zip(qs,ks)];check(scores==[400,80],'actual projected scores')
    gamma=factors(scores,100);check(gamma==[R(1,4),1],'local trigger')
    for alpha in [0,R(1,2),1]:
        transformed=[((g**alpha*q)*z)[0]*((g**(1-alpha)*k)*z)[0] for g,q,k in zip(gamma,qs,ks)]
        check(transformed==[100,80],'exponent allocation contracts product')
    gg=sp.symbols('gamma',positive=True);aa=sp.symbols('alpha',real=True)
    eq(gg**aa*gg**(1-aa),gg,'general homogeneous multiplier')
    check(factors([0,-3,100],100)==[1,1,1],'zero/negative/equality no-op')
    rejected(lambda:factors([],100),'empty observed pair set rejected')
    for bad in [sp.nan,sp.oo,-sp.oo]:rejected(lambda:factors([bad],100),'nonfinite observed score rejected')
    rejected(lambda:factors([1],0),'nonpositive threshold rejected')
    eq(scores[1]*gamma[0],20,'global factor damages healthy head')
    content=[300,20];rotary=[100,60]
    shared=[g*cc+g*rr for g,cc,rr in zip(gamma,content,rotary)]
    check(shared==[100,80],'both score paths receive exactly one local factor')
    wrong=[cc+R(1,4)*rr for cc,rr in zip(content,rotary)]
    check(wrong==[325,35],'shared-key-only counterexample')
    fault(wrong[1]!=scores[1],'healthy head changed by shared-key intervention')
    # Actual dimension-scaled scalar content/rotary vectors, not assigned scores.
    cq=[30*sp.sqrt(2),2*sp.sqrt(2)];ck=[10,10];rq=[10*sp.sqrt(2),6*sp.sqrt(2)];rk=10
    for i,g in enumerate(gamma):
        eq(((sp.sqrt(g)*cq[i])*(sp.sqrt(g)*ck[i])+(g*rq[i])*rk)/sp.sqrt(2),shared[i],'shared vectors exact contraction')
    eq(22*20*gamma[0],110,'old observation on new query endpoint')
    fault(110>100,'lagged threshold claim fails')
    eq((2*10)*(2*10),400,'changed-input next-batch endpoint')
    fault(R(1,4)*400+10!=R(1,4)*(400+10),'unscaled additive score bias')
    fault((10/abs(sp.Integer(10)))*(10/abs(sp.Integer(10)))!=R(1,4),'unit Q/K normalization cancels scaling')
    # Complete EMA-SGD + candidate-state clipping variant, with state retained.
    rows=[qs[0],ks[0],qs[1],ks[1]]
    states=[sp.zeros(1,2) for _ in rows];counts=[0]*len(rows)
    gradients=[sp.Matrix([[-30,0]]),*[sp.zeros(1,2) for _ in range(3)]]
    candidates=[]
    for i,(row,gradient) in enumerate(zip(rows,gradients)):
        states[i]=(states[i]+gradient)/2
        candidates.append(row-R(1,5)*states[i]);counts[i]+=1
    candidateq,candidatek=candidates[:2];Bq=states[0]
    eq(candidateq,sp.Matrix([[33,0]]),'candidate EMA-SGD projection')
    candidate_scores=[(candidates[i]*z)[0]*(candidates[i+1]*z)[0] for i in [0,2]]
    gammas=factors(candidate_scores,100);candidate_score=candidate_scores[0];g=gammas[0]
    final=[sp.sqrt(gammas[i//2])*row for i,row in enumerate(candidates)]
    eq((final[0]*z)[0]*(final[1]*z)[0],100,'candidate measured full controlled step')
    eq((final[2]*z)[0]*(final[3]*z)[0],80,'complete controlled step healthy head unchanged')
    eq(Bq,sp.Matrix([[-15,0]]),'moment retained rather than silently rescaled')
    check(all(state==sp.zeros(1,2) for state in states[1:]),'all three zero-gradient EMA states retained')
    check(counts==[1,1,1,1],'one actual state update per independent projection row')
    fault([400,400]==[400,400] and sp.exp(400)/(sp.exp(400)+sp.exp(400))==R(1,2),'large common score shift does not imply concentration')
    return report(dict(scores=[str(x) for x in scores],factors=[str(x) for x in gamma],candidate_score=str(candidate_score),candidate_factor=str(g),moment=[str(x) for x in Bq],update_count=counts[0],update_counts=counts))

if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
