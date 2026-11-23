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

from collections import defaultdict
from functools import lru_cache

TOKENS={'START':b'{"ok":','T_FULL':b'true}','F_FULL':b'false}', 'T':b'true','F':b'false','CLOSE':b'}','BAD':b'x','OVER':b'true}x'}
LANGUAGE=frozenset([b'{"ok":true}',b'{"ok":false}'])
COMPLETE={('START','T_FULL'),('START','F_FULL'),('START','T','CLOSE'),('START','F','CLOSE')}
EOS='EOS';FAIL='FAIL_BOUND'

def payload(history):return b''.join(TOKENS[t] for t in history if t not in [EOS,FAIL])
def viable(p):return any(s.startswith(p) for s in LANGUAGE)
def transition(prefix,token):
    if token==EOS:
        if prefix not in LANGUAGE:raise ValueError('premature EOS')
        return prefix
    result=prefix+TOKENS[token]
    if not viable(result):raise ValueError('whole-token dead transition')
    return result

def base(history):
    if not history:return {'START':R(3,4),'BAD':R(1,4)}
    if history==('START',):return dict(zip(['T_FULL','F_FULL','T','F','CLOSE','BAD','OVER'],[R(3,10),R(1,10),R(1,5),R(1,10),R(1,10),R(1,10),R(1,10)]))
    if history==('START','T'):return {'CLOSE':R(3,4),'BAD':R(1,4)}
    if history==('START','F'):return {'CLOSE':R(1,2),'BAD':R(1,2)}
    if history in COMPLETE:return {'EOS':R(3,4),'BAD':R(1,4)}
    return {'EOS':sp.S.One}

def is_legal(history,event):
    p=payload(history)
    return p in LANGUAGE if event==EOS else viable(p+TOKENS[event])

def mask(history,row):
    legal={v:p for v,p in row.items() if is_legal(history,v)}
    Z=sum(legal.values())
    if not legal or Z<=0:raise ValueError('zero legal mass')
    return {v:p/Z for v,p in legal.items() if p>0}

def target(history):return mask(history,base(history))

def proposal(history):
    if not history:return {'START':R(3,4),'BAD':R(1,4)}
    if history==('START',):return {'T':R(1,2),'F':R(1,4),'OVER':R(1,4)}
    if history==('START','T'):return {'OVER':R(1,2),'CLOSE':R(1,2)}
    if history==('START','F'):return {'CLOSE':R(1,4),'BAD':R(3,4)}
    if history in COMPLETE:return {'EOS':R(1,2),'BAD':R(1,2)}
    return {'EOS':sp.S.One}

def enumerate_base(masked=False):
    leaves={};stack=[((),sp.S.One)]
    while stack:
        h,m=stack.pop();row=target(h) if masked else base(h);eq(sum(row.values()),1,'finite row normalization')
        if len(h)>=5:raise ValueError('specified finite tree exceeded')
        for v,p in row.items():
            nxt=h+(v,)
            if v==EOS:leaves[nxt]=m*p
            else:stack.append((nxt,m*p))
    return leaves

@lru_cache(None)
def H(history):
    if history and history[-1]==EOS:return sp.S.One if payload(history) in LANGUAGE else sp.S.Zero
    return sum(p*H(history+(v,)) for v,p in base(history).items())

def correction(p,q):
    names=set(p)|set(q);accepted={v:min(p.get(v,0),q.get(v,0)) for v in names};rej=1-sum(accepted.values())
    residual={v:max(p.get(v,0)-q.get(v,0),0)/rej for v in names if p.get(v,0)>q.get(v,0)} if rej else {}
    return accepted,rej,residual

# Independently enumerate full draft chains, then all reached verification branches.
def drafts(history,k):
    if k==0:return {():sp.S.One}
    out={}
    for v,p in proposal(history).items():
        tails={():sp.S.One} if v==EOS else drafts(history+(v,),k-1)
        for tail,m in tails.items():out[(v,)+tail]=p*m
    return out

def block_iteration(history,k,budget):
    output=defaultdict(lambda:sp.S.Zero)
    def verify(draft,index,emitted,mass):
        committed=history+emitted;p=target(committed)
        if index==len(draft):
            if len(emitted)<budget and (not emitted or emitted[-1]!=EOS):
                for v,pp in p.items():output[emitted+(v,)]+=mass*pp
            else:output[emitted]+=mass
            return
        v=draft[index];q=proposal(history+draft[:index]);accept=min(sp.S.One,p.get(v,0)/q[v])
        if accept:
            nxt=emitted+(v,)
            # Explicit synchronized model/grammar/history commit.
            eq(sp.Integer(transition(payload(committed),v)==payload(history+nxt)),1,'accepted recognizer/model history aligned')
            if v==EOS or len(nxt)==budget:output[nxt]+=mass*accept
            else:verify(draft,index+1,nxt,mass*accept)
        if accept<1:
            accepted,rej,residual=correction(p,q)
            if rej<=0:raise ValueError('reachable rejection needs residual')
            # Restore to committed, not the recognizer of full tentative draft.
            restored=payload(committed)
            for y,rr in residual.items():
                updated=transition(restored,y)
                check(updated==payload(committed+(y,)),'correction advances restored prefix')
                output[emitted+(y,)]+=mass*(1-accept)*rr
    for d,m in drafts(history,min(k,budget)).items():verify(d,0,(),m)
    eq(sum(output.values()),1,'all block branches normalized')
    return dict(output)

def direct_law(history,budget):
    if history and history[-1]==EOS:return {history:sp.S.One}
    if budget==0:return {history+(FAIL,):sp.S.One}
    out=defaultdict(lambda:sp.S.Zero)
    for v,p in target(history).items():
        for end,m in direct_law(history+(v,),budget-1).items():out[end]+=p*m
    return dict(out)

def speculative_law(history,k,budget):
    if history and history[-1]==EOS:return {history:sp.S.One}
    if budget==0:return {history+(FAIL,):sp.S.One}
    out=defaultdict(lambda:sp.S.Zero)
    for emitted,m in block_iteration(history,k,budget).items():
        for end,p in speculative_law(history+emitted,k,budget-len(emitted)).items():out[end]+=m*p
    return dict(out)

def rollback_trace(stale=False):
    h=();grammar=b'';model=();provisional=[];draft=('START','T','OVER');uniforms=[R(1,2)]*3
    for i,(v,u) in enumerate(zip(draft,uniforms)):
        provisional.append(payload(draft[:i+1]));p=target(h);q=proposal(draft[:i]);alpha=min(sp.S.One,p.get(v,0)/q[v])
        if u<=alpha:
            h+=(v,);model=h;grammar=transition(grammar,v)
        else:
            if stale:grammar=provisional[-1]
            accepted,rej,residual=correction(p,q);check(residual=={'CLOSE':sp.S.One},'trace correction after accepted prefix')
            y=next(iter(residual));grammar=transition(grammar,y);h+=(y,);model=h
            if grammar!=payload(model):raise ValueError('stale recognizer/model mismatch')
            return dict(history=h,grammar=grammar,model=model,rejected=v,provisional=provisional)
    raise ValueError('trace did not exercise rejection')

def run_checks():
    z=sp.Matrix([R(2,3),R(5,6)]);readout=sp.Matrix([[R(3,2)*sp.log(3),0],[R(3,2)*sp.log(2),0],[0,0]])
    logits=readout*z;mass=[sp.exp(v) for v in logits];response=[v/sum(mass) for v in mass]
    check(response==[R(1,2),R(1,3),R(1,6)],'independent response readout')
    retained=response[0]+response[2];check([response[0]/retained,0,response[2]/retained]==[R(3,4),0,R(1,4)],'first response mask')
    base_leaves=enumerate_base();local=enumerate_base(True);check((len(base_leaves),len(local))==(14,4),'entire specified terminated trees')
    eq(sum(base_leaves.values()),1,'base tree mass');eq(sum(local.values()),1,'local tree mass')
    valid={h:p for h,p in base_leaves.items() if payload(h) in LANGUAGE};accepted=sum(valid.values());eq(accepted,R(27,80),'base accepted mass')
    paths=[('START','T_FULL',EOS),('START','F_FULL',EOS),('START','T','CLOSE',EOS),('START','F','CLOSE',EOS)]
    check([valid[h] for h in paths]==[R(27,160),R(9,160),R(27,320),R(9,320)],'all accepted base path masses')
    check([local[h] for h in paths]==[R(3,7),R(1,7),R(2,7),R(1,7)],'all locally masked path masses')
    conditioned={h:p/accepted for h,p in valid.items()};check([conditioned[h] for h in paths]==[R(1,2),R(1,6),R(1,4),R(1,12)],'globally conditioned token paths')
    true=lambda law:sum(p for h,p in law.items() if payload(h)==b'{"ok":true}')
    eq(true(local),R(5,7),'local true-object mass');eq(true(conditioned),R(3,4),'global true-object mass')
    eq(H(()),accepted,'independent finite H recursion')
    histories=set(h[:i] for h in base_leaves for i in range(len(h)))
    for h in histories:
        if H(h)>0:
            globally_next={v:p*H(h+(v,))/H(h) for v,p in base(h).items()}
            eq(sum(globally_next.values()),1,'globally conditioned next-event normalized')
    check(len(histories)>4,'H checked at every supported finite-tree history')
    check(is_legal(('START',),'T_FULL'),'multi-byte token can cross grammar boundary')
    fault(not is_legal(('START',),'OVER'),'whole-token overrun rejected')
    fault(viable(b'{"ok":true}') and not viable(b'{"ok":true}x'),'first-byte or first-acceptance-only matching fails')
    rejected(lambda:transition(b'{"ok":',EOS),'premature EOS rejected')
    rejected(lambda:mask((),{'BAD':sp.S.One}),'positive legal set with zero mass rejected')
    rejected(lambda:mask(('BAD',),base(('BAD',))),'dead prefix empty legal set rejected')
    unavailable_prefix=b'a';vocab=[b'a',b'x'];L={b'ab'}
    check(any(s.startswith(unavailable_prefix) for s in L),'byte prefix viable')
    fault(not any(s.startswith(unavailable_prefix+v) for s in L for v in vocab),'viable suffix unavailable in token vocabulary')
    L1={b'ab'};L2={b'ac'};check(all(any(s.startswith(b'a') for s in Lx) for Lx in [L1,L2]),'individual viable prefixes')
    fault(not (L1&L2),'empty joint language despite individually viable prefix')
    raw={'BAD':R(3,5),'A':R(3,10),'B':R(1,10)};legal={'A','B'}
    fault(max(raw,key=raw.get) not in legal and max({v:p for v,p in raw.items() if v in legal},key=raw.get)=='A','top1 before versus after grammar order')
    for q in [{'A':R(1,4),'B':R(1,2),'C':R(1,4)},{'A':sp.S.One}]:
        p={'A':R(3,4),'C':R(1,4)};a,j,r=correction(p,q)
        for v in set(p)|set(q):eq(a[v]+j*r.get(v,0),p.get(v,0),'exact constrained one-token mixture')
    fault(R(3,4)/sp.S.One!=R(3,4)/R(1,2),'raw draft denominator after masking changes acceptance')
    configurations=0
    for k in [1,2,3]:
        for budget in [1,2,3,4,5]:
            oracle=direct_law((),budget);actual=speculative_law((),k,budget)
            check(set(oracle)==set(actual),'speculative complete support including bound failures')
            for end,m in oracle.items():eq(actual[end],m,'independent complete constrained block law')
            eq(sum(actual.values()),1,'complete stopped speculative law');configurations+=1
    trace=rollback_trace();check(trace['history']==('START','T','CLOSE'),'rejection after two accepted tokens corrected')
    check(trace['grammar'] in LANGUAGE and trace['model']==trace['history'],'matched final state triple')
    rejected(lambda:rollback_trace(stale=True),'stale recognizer after rejection detected')
    small=direct_law((),2);failure=sum(m for h,m in small.items() if h[-1]==FAIL)
    eq(failure,1,'hard-bound failure mass retained even for complete payload without EOS')
    fault(sum(m for h,m in small.items() if h[-1]==EOS)!=1,'omitting hard-bound failure mass loses normalization')
    return report(dict(base_leaves=[{'history':list(h),'mass':str(m)} for h,m in sorted(base_leaves.items())],local_leaves=[{'history':list(h),'mass':str(m)} for h,m in sorted(local.items())],accepted_mass=str(accepted),true_local=str(true(local)),true_conditioned=str(true(conditioned)),block_configurations=configurations,H_histories=len(histories),rollback_history=list(trace['history'])))

if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
