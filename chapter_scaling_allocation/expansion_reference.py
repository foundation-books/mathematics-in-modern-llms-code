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
def rejected(fn, label, message=None):
    try:
        fn()
    except ValueError as exc:
        fault(message is None or str(exc) == message, label)
    else:
        raise AssertionError(label)
def report(data):
    return dict(python=platform.python_version(), sympy=sp.__version__,
                checks=len(CHECKS), fault_checks=FAULTS, **data)

from itertools import product
from expansion_student import (EVALUATION_DOCUMENT_COUNTS,
                               EVALUATION_PAIRED_COUNTS, Q_MISSING)


def mean_statistics(values):
    """Mean, sample variance and squared estimated SE for declared iid units.

    The calculation does not establish that the supplied units are iid.
    """
    values = tuple(map(sp.Rational, values))
    if len(values) < 2:
        raise ValueError("Sample variance needs at least two observations")
    mean = sum(values) / len(values)
    variance = sum((value - mean)**2 for value in values) / (len(values) - 1)
    return mean, variance, variance / len(values)


def evaluation_sampling():
    """F1(b): item/document models, conditional test and fixed family."""
    counts = EVALUATION_PAIRED_COUNTS
    differences = [value for value, count in zip((0, 1, -1, 0), counts)
                   for _ in range(count)]
    item_mean, item_variance, item_se2 = mean_statistics(differences)
    document_means = tuple(R(row[1] - row[2], sum(row))
                           for row in EVALUATION_DOCUMENT_COUNTS)
    doc_mean, doc_variance, doc_se2 = mean_statistics(document_means)
    discordant = counts[1] + counts[2]
    iid_item_pvalue = min(sp.S.One, 2 * sum(
        sp.binomial(discordant, j) for j in range(min(counts[1:3]) + 1)
    ) / sp.Integer(2)**discordant)
    family_size = sp.binomial(5, 2)
    # The p-value belongs only to the fixed-sample iid paired-binary model.
    # Bonferroni needs valid tests for the ten prespecified comparisons;
    # dependence among those tests is permitted.
    return dict(item_n=len(differences), item_mean=item_mean,
                item_variance=item_variance, item_se2=item_se2,
                document_n=len(document_means), document_means=document_means,
                document_mean=doc_mean, document_variance=doc_variance,
                document_se2=doc_se2, iid_item_exact_pvalue=iid_item_pvalue,
                family_size=family_size, local_threshold=R(1, 20)/family_size)

def importance(p,q):
    if len(p)!=len(q) or sum(p)!=1 or sum(q)!=1 or any(x<0 for x in [*p,*q]):raise ValueError('laws')
    if any(pp>0 and qq==0 for pp,qq in zip(p,q)):raise ValueError('unsupported objective')
    return [pp/qq if qq else sp.S.Zero for pp,qq in zip(p,q)]
def acquire(priorities,fallback):
    if any(x<0 for x in priorities):raise ValueError('priority')
    total=sum(priorities)
    if total:return [sp.sympify(x)/total for x in priorities]
    if len(fallback)!=len(priorities) or sum(fallback)!=1 or any(x<=0 for x in fallback):raise ValueError('fallback')
    return list(fallback)

def run_checks():
    p=[R(1,3)]*3;q=[R(1,15),R(1,3),R(3,5)];f=[sp.S.One,sp.S.Zero,-sp.S.One]
    w=importance(p,q);check(w==[5,1,R(5,9)],'objective/acquisition weights')
    mu=sum(pp*ff for pp,ff in zip(p,f));raw=sum(qq*ff for qq,ff in zip(q,f));corrected=sum(qq*ww*ff for qq,ww,ff in zip(q,w,f))
    eq(mu,0,'target mean');eq(raw,-R(8,15),'raw acquisition mean');eq(corrected,mu,'finite change of measure')
    vector_values=[sp.Matrix([value,value**2]) for value in f]
    vector_target=sum((pp*value for pp,value in zip(p,vector_values)),sp.zeros(2,1))
    vector_corrected=sum((qq*ww*value for qq,ww,value in zip(q,w,vector_values)),sp.zeros(2,1))
    eq(vector_target,[0,R(2,3)],'declared vector integrand target')
    eq(vector_corrected,vector_target,'coordinatewise acquisition correction')
    variance=sum(pp**2*ff**2/qq for pp,ff,qq in zip(p,f,q))-mu**2
    eq(variance,R(50,27),'exact corrected one-draw variance');eq(sum(pp*ff**2 for pp,ff in zip(p,f)),R(2,3),'direct variance')
    n=sp.symbols('n',integer=True,positive=True);eq(variance/n,R(50,27)/n,'fixed IID sample mean variance')
    floor=[R(7,10)*qq+R(3,10)*pp for pp,qq in zip(p,q)];check(floor==[R(11,75),R(1,3),R(13,25)],'support mixture')
    check(min(floor)==R(11,75),'actual floor exceeds guaranteed one tenth')
    clipped=sum(qq*min(ww,2)*ff for qq,ww,ff in zip(q,w,f));eq(clipped,-R(1,5),'clipped expectation')
    fault(clipped!=mu,'weight clipping changes objective expectation')
    fault(raw!=mu,'one-draw self-normalization returns biased unweighted f')
    rejected(lambda: importance(p, Q_MISSING), 'D normalized row lacks target support',
             'unsupported objective')
    rejected(lambda: importance(p, (0, *q[1:])), 'D malformed row fails normalization',
             'laws')
    check(acquire([0,0,0],p)==p,'zero-priority full-support fallback')
    rejected(lambda:acquire([0,0,0],[1,0,0]),'invalid fallback rejected')
    outcomes = []
    joint = {}
    for i, j in product(range(3), repeat=2):
        q2 = q if i == 0 else q[::-1]
        mass = p[i] * q2[j]
        weight = importance(p, q2)[j]
        z2 = weight * f[j]
        raw_average = (f[i] + f[j]) / 2
        corrected_average = (f[i] + z2) / 2
        joint[i, j] = mass, z2, raw_average, corrected_average
        outcomes.append(dict(first=i+1, second=j+1, conditional_row=list(map(str, q2)),
                             proposal=str(q2[j]), weight=str(weight), joint_mass=str(mass),
                             z2=str(z2), raw_average=str(raw_average),
                             corrected_average=str(corrected_average)))
    eq(sum(mass for mass, *_ in joint.values()), 1, 'E nine-outcome law mass')
    mean_raw = sum(mass * raw_avg for mass, _, raw_avg, _ in joint.values())
    mean_corrected = sum(mass * avg for mass, _, _, avg in joint.values())
    eq(mean_raw, R(4, 45), 'E unconditional raw average')
    eq(mean_corrected, mu, 'E unconditional corrected average')
    z2_means, average_means = [], []
    for i in range(3):
        row = [joint[i, j] for j in range(3)]
        first_mass = sum(mass for mass, *_ in row)
        eq(first_mass, p[i], f'E first-task marginal {i+1}')
        z2_means.append(sum(mass * z2 for mass, z2, _, _ in row) / first_mass)
        average_means.append(sum(mass * avg for mass, _, _, avg in row) / first_mass)
        eq(z2_means[i], mu, f'E second contribution given first task {i+1}')
        eq(average_means[i], (f[i] + mu) / 2, f'E corrected average given first task {i+1}')
    eq(sum(pp * avg for pp, avg in zip(p, average_means)), mean_corrected,
       'E iterated expectation for the corrected average')
    future_mass=sum(p[i]*(q if i==0 else q[::-1])[0] for i in range(3))
    future_selected=[p[i]*(q if i==0 else q[::-1])[0]/future_mass for i in range(3)]
    check(future_selected==[R(1,19),R(9,19),R(9,19)],'posterior first task after a future selection')
    fault(sum(pp*ff for pp,ff in zip(future_selected,f))==-R(8,19),'future-selected law cannot use unfiltered saved proposal')
    # Scalar oracle minimizes the second moment; its signed-integrand variance here is nonzero, and full support cannot attain the infimum.
    oracle=[R(1,2),0,R(1,2)];second=sum(p[i]**2*f[i]**2/oracle[i] for i in [0,2])
    eq(second,R(4,9),'known scalar integrand oracle second moment')
    eq(sum(pp**2*sp.S.Zero/qq for pp,qq in zip(p,q)),0,'zero integrand any supported proposal gives zero second moment')
    check(second<sum(p[i]**2*f[i]**2/q[i] for i in range(3)),'oracle is specific to known integrand')
    evaluation = evaluation_sampling()
    eq(evaluation['item_mean'], R(1,5), 'paired evaluation mean')
    eq(evaluation['item_variance'], R(36,95), 'paired difference sample variance')
    eq(evaluation['item_se2'], R(9,475), 'iid item squared estimated SE')
    eq(evaluation['document_mean'], R(1,5), 'document evaluation mean')
    eq(evaluation['document_variance'], R(2,15), 'document mean sample variance')
    eq(evaluation['document_se2'], R(1,30), 'iid document squared estimated SE')
    eq(evaluation['iid_item_exact_pvalue'], R(37,128), 'iid item exact test only')
    eq(evaluation['family_size'], 10, 'prespecified all-pairs family')
    eq(evaluation['local_threshold'], R(1,200), 'Bonferroni local level')
    return report(dict(target_mean=str(mu), raw_mean=str(raw), variance=str(variance),
                       clipped_mean=str(clipped), adaptive_outcomes=outcomes,
                       adaptive_raw=str(mean_raw), adaptive_corrected=str(mean_corrected),
                       conditional_second_contribution=list(map(str, z2_means)),
                       conditional_corrected_average=list(map(str, average_means)),
                       evaluation_sampling={"part": "F1(b) Evaluation sampling",
                                            **{key: str(value) for key, value in evaluation.items()}}))

if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
