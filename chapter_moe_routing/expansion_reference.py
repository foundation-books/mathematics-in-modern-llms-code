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
def rejected(fn, label, expected=None):
    try: fn()
    except ValueError as error:
        if expected is not None and str(error) != expected:
            raise AssertionError(f"{label}: wrong rejection reason: {error}") from error
        fault(True, label)
    else: raise AssertionError(label)
def report(data):
    return dict(python=platform.python_version(), sympy=sp.__version__,
                checks=len(CHECKS), fault_checks=FAULTS, **data)

from collections import Counter
from dataclasses import dataclass
from itertools import product

@dataclass(frozen=True)
class Route:
    token: int
    slot: int
    expert: int
    coefficient: sp.Expr
    version: int = 0
    @property
    def key(self): return (self.token, self.slot)

def compositions(total, n):
    if n == 1:
        yield (total,); return
    for first in range(total+1):
        for tail in compositions(total-first,n-1): yield (first,*tail)

def fill(routes, homes, devices):
    if type(devices) is not int or devices < 1: raise ValueError('devices')
    if len(homes) % devices or Counter(homes.values()) != Counter({r:len(homes)//devices for r in range(devices)}):
        raise ValueError('equal expert homes required')
    if len(routes) % devices: raise ValueError('nondivisible total')
    if len({r.key for r in routes}) != len(routes): raise ValueError('duplicate route key')
    slots=[[] for _ in range(devices)]
    for r in sorted(routes,key=lambda r:(r.expert,r.token,r.slot)): slots[homes[r.expert]].append(r)
    target=len(routes)//devices; frozen={r for r in range(devices) if len(slots[r])==target}; trace=[]
    while len(frozen)<devices:
        under=[r for r in range(devices) if len(slots[r])<target]
        if not under: break
        u=under[0]; v=next(r for r in range(devices) if len(slots[r])>target)
        need=target-len(slots[u]); before=len(slots[v])
        check(all(homes[t.expert]==v for t in slots[v]),'unfrozen donor has only home routes')
        moved=slots[v][:need];slots[u].extend(moved);del slots[v][:need];frozen.add(u)
        if len(slots[v])==target:frozen.add(v)
        trace.append((u,v,need,before-target))
    return slots,trace

def remote_types(slots,homes):
    return [len({r.expert for r in slot if homes[r.expert]!=d}) for d,slot in enumerate(slots)]

def allocation_oracle(counts,homes,devices):
    if sum(counts)%devices:raise ValueError('nondivisible total')
    target=sum(counts)//devices;best=None;cases=0
    for rows in product(*(list(compositions(c,devices)) for c in counts)):
        if any(sum(row[r] for row in rows)!=target for r in range(devices)):continue
        score=max(sum(row[r]>0 and homes[e+1]!=r for e,row in enumerate(rows)) for r in range(devices))
        best=score if best is None else min(best,score);cases+=1
    return best,cases

def fixture():
    z=sp.Matrix([R(2,3),R(5,6)]);c=sp.Matrix([R(3,2),0])
    pairs=[(1,2)]*4+[(1,3)]*2+[(1,4),(2,3),(2,4),(3,4)]
    routes=[Route(t,s,e,R(s+1,3)) for t,p in enumerate(pairs,1) for s,e in enumerate(p)]
    features={t:z+(t-1)*sp.Matrix([R(1,10),-R(1,20)]) for t in range(1,11)}
    homes={1:0,2:0,3:1,4:1};destination={r.key:1 if r.expert==1 and r.token<=3 else homes[r.expert] for r in routes}
    return routes,features,c,homes,destination

def run_checks():
    routes,x,c,homes,dest=fixture();theta=sp.symbols('theta1:5',real=True)
    outputs=[sum(r.coefficient*theta[r.expert-1]*(c.T*x[t])[0] for r in routes if r.token==t) for t in range(1,11)]
    loss=sum(outputs);grad=sp.Matrix([sp.diff(loss,t) for t in theta]);values=dict(zip(theta,[1,2,3,4]))
    eq(loss.subs(values),R(431,10),'probe');eq(grad,[R(203,60),R(281,60),R(263,60),R(43,10)],'direct parameter derivatives')
    matrix=sp.zeros(4,2)
    for r in routes:matrix[r.expert-1,dest[r.key]]+=1
    eq(matrix,sp.Matrix([[4,3],[6,0],[0,4],[0,3]]),'route allocation')
    eq(matrix*sp.ones(2,1),[7,6,4,3],'row conservation');eq(matrix.T*sp.ones(4,1),[10,10],'column balance')
    local=[sum(r.coefficient*theta[r.expert-1]*(c.T*x[r.token])[0] for r in routes if dest[r.key]==d) for d in range(2)]
    eq(sum(local),loss,'placed scalar equality')
    returned=[sp.Matrix([sp.diff(l,t) for t in theta]) for l in local]
    eq(sum(returned,sp.zeros(4,1)),grad,'owner-correct return')
    remote=sum(r.coefficient*(c.T*x[r.token])[0] for r in routes if dest[r.key]!=homes[r.expert]);eq(remote,R(23,20),'remote owner gradient')
    stale_outputs=[o+(c.T*x[t])[0]/3 if t<=3 else o for t,o in enumerate(outputs,1)]
    eq(sum(stale_outputs)-loss,R(23,20),'stale-copy forward change')
    fault(stale_outputs!=outputs,'stale replica changes routewise outputs')
    fault(returned[0]!=grad,'omitted remote gradient return')
    swapped=sum((R(2,3)*theta[0]+R(1,3)*theta[1])*(c.T*x[1])[0] for _ in range(1))
    fault(sp.simplify(swapped-outputs[0])!=0,'coefficient swap preserves counts but changes map')
    rejected(lambda:fill(routes[:-1]+[routes[0]],homes,2),'duplicated key rejected','duplicate route key')
    fault(Counter(r.key for r in routes[:-1])!=Counter(r.key for r in routes),'omitted route detected')
    # Independent free coefficient/input differentiation, then normalization chain.
    av=sp.symbols('a0:20',real=True); xv=sp.symbols('x0:20',real=True);XX=sp.Matrix(10,2,xv)
    direct=sum(av[i]*theta[r.expert-1]*(c.T*XX[r.token-1,:].T)[0] for i,r in enumerate(routes))
    for i,r in enumerate(routes):
        eq(sp.diff(direct,av[i]),theta[r.expert-1]*(c.T*XX[r.token-1,:].T)[0],'coefficient derivative')
    for t in range(1,11):
        expected=sum(av[i]*theta[r.expert-1] for i,r in enumerate(routes) if r.token==t)*c
        eq(sp.Matrix([sp.diff(direct,v) for v in XX[t-1,:]]),expected,'input derivative')
    s1,s2=sp.symbols('s1 s2',positive=True); v1,v2=sp.symbols('v1 v2',real=True)
    norm=(s1*v1+s2*v2)/(s1+s2)
    for ss,v in [(s1,v1),(s2,v2)]:eq(sp.diff(norm,ss),(v-norm)/(s1+s2),'normalized coefficient chain')
    histograms=0
    for experts,devices,total in [(4,2,8),(4,4,8),(6,3,6)]:
        h={e+1:e//(experts//devices) for e in range(experts)}
        for counts in compositions(total,experts):
            rr=[Route(e*100+j,0,e+1,sp.S.One) for e,n in enumerate(counts) for j in range(n)]
            slots,tr=fill(rr,h,devices);histograms+=1
            check([len(slot) for slot in slots]==[total//devices]*devices,'histogram column balance')
            check(Counter(r for slot in slots for r in slot)==Counter(rr),'histogram exact route multiset')
            check(len(tr)<=devices-1,'histogram termination')
            check(max(remote_types(slots,h))<=experts//devices,'histogram type upper bound')
    check(histograms==792,'complete prescribed histogram domains')
    crossing=[Route(e*100+j,0,e+1,1) for e,n in enumerate([0,0,3,3,3,3]) for j in range(n)]
    slots,tr=fill(crossing,{e+1:e//2 for e in range(6)},3)
    check(any(need>surplus for u,v,need,surplus in tr),'full deficit crosses donor surplus')
    for rr,h,d in [([],homes,2),([Route(i,0,1,1) for i in range(3)],{1:0},1)]:
        slots,tr=fill(rr,h,d);check(len(tr)==0,'no-work / one-device direct cases')
    rejected(lambda:fill(routes[:-1],homes,2),'nondivisible target rejected')
    rejected(lambda:fill(routes,{1:0,2:0,3:0,4:1},2),'uneven theorem home partition rejected')
    lower=sp.zeros(8,4)
    for e in range(2,8):lower[e,e//2]=4
    for e,src,to,n in [(2,1,0,4),(3,1,0,2),(4,2,1,4),(6,3,2,2)]:lower[e,src]-=n;lower[e,to]+=n
    eq(lower.T*sp.ones(8,1),[6]*4,'lower witness loads');check(sp.ceiling(R(6,4))==2,'lower counting proof witness')
    check(max(sum(int(bool(lower[e,r]>0))*(e//2!=r) for e in range(8)) for r in range(4))==2,'matching maximum remote types')
    oracle_histograms=0;feasible_allocations=0
    for hist in compositions(4,4):
        best,cases=allocation_oracle(hist,homes,2);oracle_histograms+=1;feasible_allocations+=cases
        rr=[Route(e*100+j,0,e+1,1) for e,n in enumerate(hist) for j in range(n)]
        slots,tr=fill(rr,homes,2);check(best<=max(remote_types(slots,homes))<=2,'exact optimum versus constructor')
    return report(dict(histograms=histograms,oracle_histograms=oracle_histograms,feasible_allocations=feasible_allocations,outputs=[str(o.subs(values)) for o in outputs],probe=str(loss.subs(values)),gradient=[str(t) for t in grad],remote_gradient=str(remote)))

if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
