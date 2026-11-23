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

def relu(x):return x.applyfunc(lambda t:max(t,0))
def weighted_gradient(numerators,masses):
    if len(numerators)!=len(masses) or any(m<0 for m in masses) or sum(masses)<=0:raise ValueError('normalization mass')
    return sum(numerators,sp.zeros(*numerators[0].shape))/sum(masses)

def run_checks():
    z=sp.Matrix([R(2,3),R(5,6)]);W1=sp.Matrix([[1,2],[3,4],[-1,1],[2,0]]);W2=sp.Matrix([[1,0,2,-1],[0,1,1,2]])
    pre=W1*z;h=relu(pre);y=W2*h;g=sp.Matrix([1,2])
    check(all(x>0 for x in pre),'specified active ReLU region')
    eq(h,[R(7,3),R(16,3),R(1,6),R(4,3)],'complete hidden coordinates');eq(y,[R(4,3),R(49,6)],'serial output')
    partials=[W2[:,:2]*h[:2,:],W2[:,2:]*h[2:,:]]
    eq(partials[0],[R(7,3),R(16,3)],'first output partial');eq(partials[1],[-1,R(17,6)],'second output partial');eq(sum(partials,sp.zeros(2,1)),y,'sum reconstructs output')
    wrong_h=relu(W1[:,0]*z[0])+relu(W1[:,1]*z[1]);wrong=W2*wrong_h
    eq(wrong,[R(8,3),R(53,6)],'premature ReLU witness');fault(wrong!=y,'nonlinear-before-sum schedule fails')
    barh=W2.T*g;barz=W1.T*barh;eq(barh,[1,2,4,3],'hidden adjoint');eq(barz,[9,14],'input adjoint')
    input_partials=[W1[:2,:].T*barh[:2,:],W1[2:,:].T*barh[2:,:]]
    eq(input_partials[0],[7,10],'input gradient use0');eq(input_partials[1],[2,4],'input gradient use1')
    a=sp.symbols('a0:8',real=True);b=sp.symbols('b0:8',real=True);x=sp.symbols('x0:2',real=True)
    AA=sp.Matrix(4,2,a);BB=sp.Matrix(2,4,b);XX=sp.Matrix(x);probe=(g.T*BB*AA*XX)[0];subs=dict(zip(a,W1));subs.update(zip(b,W2));subs.update(zip(x,z))
    grad1=sp.Matrix(4,2,[sp.diff(probe,v).subs(subs) for v in a]);grad2=sp.Matrix(2,4,[sp.diff(probe,v).subs(subs) for v in b]);inputg=sp.Matrix([sp.diff(probe,v).subs(subs) for v in x])
    eq(grad1,barh*z.T,'all W1 derivatives');eq(grad2,g*h.T,'all W2 derivatives');eq(inputg,barz,'independent input derivative')
    newW1=W1-grad1/10;newW2=W2-grad2/10
    shardW1=(W1[:2,:]-grad1[:2,:]/10).col_join(W1[2:,:]-grad1[2:,:]/10)
    shardW2=(W2[:,:2]-grad2[:,:2]/10).row_join(W2[:,2:]-grad2[:,2:]/10)
    eq(shardW1,newW1,'all updated W1 coordinates reconstructed')
    eq(shardW2,newW2,'all updated W2 coordinates reconstructed')
    eq(shardW2*relu(shardW1*z),newW2*relu(newW1*z),'next forward from the complete updated state')
    fault(input_partials[0]!=barz,'forgotten input-gradient contribution');fault(2*grad1!=grad1,'double logical loss seed')
    # Explicit matrices prove every Euclidean adjoint by transpose, including literal allreduce.
    I=sp.eye(2);broadcast=I.col_join(I);summation=I.row_join(I)
    check(broadcast.T==summation,'broadcast adjoint is sum');check(summation.T==broadcast,'sum adjoint is broadcast')
    ar=broadcast*summation;check(ar.T==ar,'literal allreduce adjoint')
    concatenation=sp.eye(4);slices=[concatenation[:2,:],concatenation[2:,:]]
    eq(slices[0].T*slices[0]+slices[1].T*slices[1],sp.eye(4),'ordered split/concatenate adjoint reconstruction')
    v=sp.Matrix([1,2]);w=sp.Matrix([3,4,5,6]);eq((broadcast*v).dot(w),v.dot(broadcast.T*w),'inner product broadcast adjoint')
    centers=[sp.Matrix([9,17])]+[sp.Matrix([5,13])]*3;ww=sp.Matrix(sp.symbols('w0:2',real=True));loss=sum((ww-cc).dot(ww-cc)/2 for cc in centers)/4
    direct=sp.Matrix([sp.diff(loss,v) for v in ww]).subs(dict(zip(ww,[10,20])))
    g0=sp.Matrix([1,3]);g1=sp.Matrix([5,7]);globalg=weighted_gradient([g0,3*g1],[1,3])
    eq(globalg,direct,'direct global loss matches aggregation');eq(globalg,[4,6],'unequal-mass global gradient');update=sp.Matrix([10,20])-globalg/10;eq(update,[R(48,5),R(97,5)],'one SGD update')
    fault((g0+g1)/2!=globalg,'mean of device means');fault(globalg/4!=globalg,'double objective division')
    eq(weighted_gradient([g0,g1,2*g1,sp.zeros(2,1)],[1,1,2,0]),globalg,'microbatches plus zero-mass rank')
    rejected(lambda:weighted_gradient([sp.zeros(2,1)],[0]),'entire zero-mass objective rejected')
    # Parameter-dependent weights and a nonseparable mean statistic.
    t=sp.symbols('t',positive=True);ells=[t*t,3*t];weights=[t,sp.S.One];den=sum(weights);weighted=sum(w*l for w,l in zip(weights,ells))/den
    rule=sum(w*sp.diff(l,t)+(l-weighted)*sp.diff(w,t) for w,l in zip(weights,ells))/den
    eq(sp.diff(weighted,t),rule,'parameter-weight quotient rule')
    r=[t,2*t+1];mu=sum(r)/2;coupled=mu**2
    eq(sp.diff(coupled,t),sum(2*mu*sp.diff(v,t)/2 for v in r),'coupled global mean gradient')
    fault(sp.simplify(coupled-sum(v*v for v in r)/2)!=0,'averaging local squared means changes coupled loss')
    MH=sp.Matrix([[3,0],[1,1]]);Q=sp.Matrix([[4,-1],[1,4]])/sp.sqrt(17);Qr=sp.Matrix([[1,0],[1/sp.sqrt(2),1/sp.sqrt(2)]])
    eq(Q.T*Q,sp.eye(2),'whole matrix polar orthogonal');S=Q.T*MH;eq(S,S.T,'whole magnitude symmetric');check(S[0,0]>0 and S.det()>0,'whole magnitude positive definite')
    fault(Q!=Qr,'independent storage-shard polar changes transform')
    eq(MH[:1,:].col_join(MH[1:,:]),MH,'gather restores coupling block');eq(Q[:1,:].col_join(Q[1:,:]),Q,'slice redistributes transformed block')
    # A complete stateful one-owner EMA-SGD step on two gradient uses.
    old=sp.Matrix([10,20]);oldstate=sp.Matrix([2,4]);owner_count=3
    B=(oldstate+globalg)/2;candidate=old-B/10;owner_count+=1
    eq(candidate,[R(97,10),R(39,2)],'complete owner state update');eq(B,[3,5],'complete owner EMA')
    check(owner_count==4,'actual one-owner next count')
    fault(globalg/2!=B,'different initial optimizer moment changes complete step despite identical gradient')
    stale=W1.copy();stale[0,0]+=1;fault(stale*z!=pre,'stale parameter replay')
    v=sp.S.One;single=R(9,10)*v-R(1,10)*(1+2);twice=R(9,10)*(R(9,10)*v-R(1,10))-R(1,10)*2
    eq(single,R(3,5),'one logical decayed update');eq(twice,R(13,25),'two erroneous state updates');fault(single!=twice,'parameter updated twice for two uses')
    # Exact route-gradient return from Chapter12's independently specified fixture.
    pairs=[(1,2)]*4+[(1,3)]*2+[(1,4),(2,3),(2,4),(3,4)];owner=sp.zeros(4,1);remote=sp.zeros(4,1)
    for i,pair in enumerate(pairs,1):
        adapt=1+R(3,20)*(i-1)
        for slot,e in enumerate(pair):
            gg=R(slot+1,3)*adapt;owner[e-1]+=gg
            if e==1 and i<=3:remote[e-1]+=gg
    eq(owner,[R(203,60),R(281,60),R(263,60),R(43,10)],'expert owner gradients');eq(remote,[R(23,20),0,0,0],'temporary expert return');fault(owner+remote!=owner,'duplicated expert return')
    return report(dict(hidden=[str(x) for x in h],output=[str(x) for x in y],W1_gradient=[[str(grad1[i,j]) for j in range(2)] for i in range(4)],W2_gradient=[[str(grad2[i,j]) for j in range(4)] for i in range(2)],input_gradient=[str(x) for x in barz],global_gradient=[str(x) for x in globalg],SGD=[str(x) for x in update],owner_EMA=[str(x) for x in B],owner_candidate=[str(x) for x in candidate],owner_count=owner_count))

if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
