import numpy as np
import math
import jax.numpy as jnp
import jax
import jax.random as rnd
from cancellations.utilities import config as cfg,tracking
from collections import deque
from inspect import signature
from jax.numpy import tanh
from jax.nn import softplus

####################################################################################################

class RunningAvg:
    def __init__(self,k):
        self.k=k
        self.recenthist=deque([])
        self._sum_=0
        self._sqsum_=0
        self.i=0

    def update(self,val,thinning=1):
        if self.i%thinning==0: self.do_update(val)
        return self.avg()

    def do_update(self,val):    
        self.i+=1
        self._sum_+=val
        self._sqsum_+=val**2
        self.recenthist.append(val)
        if len(self.recenthist)>self.k:
            self._sum_-=self.recenthist.popleft()

    def avg(self):
        return self.sum()/self.actualk()    

    def var(self,val=None,**kw):
        if val is not None: self.update(val,**kw)
        return self.sqsum()/self.actualk()-self.avg()**2

    def actualk(self):
        return len(self.recenthist)

    def sum(self): return self._sum_
    def sqsum(self): return self._sqsum_

class InfiniteRunningAvg(RunningAvg):
    def __init__(self):
        self._sum_=0
        self._sqsum_=0
        self.i=0

    def do_update(self,val):    
        self.i+=1
        self._sum_+=val
        self._sqsum_+=val**2

    def actualk(self): return self.i


def ispoweroftwo(n):
    pattern=re.compile('10*')
    return pattern.fullmatch('{0:b}'.format(n))


class ExpRunningAverage(InfiniteRunningAvg):
    def __init__(self):
        self.blocksums=[]
        self.intervals=[]
        self.i=0

    def do_update(self,val):
        if ispoweroftwo(self.i) or self.i==0:
            self.blocksums.append(InfiniteRunningAvg())
            self.intervals.append([self.i,self.i])
        self.blocksums[-1].do_update(val)
        self.intervals[-1][-1]+=1
        self.i+=1

    def sum(self):
        return sum([e.sum() for e in self.blocksums])

    def sqsum(self): return sum([e.sqsum() for e in self.blocksums])

    def avg(self):
        if self.i<=1: return None
        prevlen=self.intervals[-2][1]-self.intervals[-2][0]
        curlen=self.intervals[-1][1]-self.intervals[-1][0]
        return (self.blocksums[-1].sum()+self.blocksums[-2].sum())/(prevlen+curlen)


####################################################################################################

@jax.jit
def sqloss(Y1,Y2):
    Y1,Y2=[jnp.squeeze(_) for _ in (Y1,Y2)]
    return jnp.average(jnp.square(Y1-Y2))


@jax.jit
def dot(Y1,Y2):
    #Y1,Y2=[jnp.squeeze(_) for _ in (Y1,Y2)]
    Y1,Y2=[jnp.atleast_1d(jnp.squeeze(_)) for _ in (Y1,Y2)]
    n=Y1.shape[0]
    return jnp.dot(Y1,Y2)/n


@jax.jit
def SI_loss(Y,Y_target):
    return 1-dot(Y,Y_target)**2/(dot(Y,Y)*dot(Y_target,Y_target))

@jax.jit
def log_SI_loss(Y,Y_target):
    Y,Y_target=[jnp.squeeze(_) for _ in (Y,Y_target)]
    return jnp.log(dot(Y_target,Y_target))+jnp.log(dot(Y,Y))-2*jnp.log(dot(Y,Y_target))


def overlap(Y1,Y2,weights):
    vec=weights*Y1*Y2
    assert(weights.shape==Y1.shape)
    assert(Y1.shape==Y2.shape)
    return jnp.sum(vec)

@jax.jit
def weighted_SI_loss(Y,Y_target,relweights):
    return 1-overlap(Y,Y_target,relweights)**2/(overlap(Y,Y,relweights)*overlap(Y_target,Y_target,relweights))


@jax.jit
def prod(L):
    out=1
    for array in L:
        out*=array
    return out


def swap(x,y):
    return (y,x)


def gen_nd_gaussian_density(var):
    def density(X):
        n=X.shape[-2]
        d=X.shape[-1]
        normalization=1/math.sqrt(2*math.pi*var)**(n*d)
        return normalization*jnp.exp(-jnp.sum(X**2,axis=(-2,-1))/(2*var))
    return density



@jax.jit
def ReLU(x):
    return jnp.maximum(x,0) 

@jax.jit
def DReLU(x):
    return jnp.minimum(jnp.maximum(x,-1),1)

drelu=DReLU

@jax.jit
def leaky_ReLU(x):
    return jnp.maximum(x,.01*x)


sigmoid=jax.jit(lambda x:(jnp.tanh(x)+1)/2)
slowsigmoid_odd=jax.jit(lambda x: x/jnp.sqrt(1+x**2))
slowsigmoid_01=jax.jit(lambda x: (slowsigmoid_odd(x)+1)/2)




ac_aliases={\
    'ReLU':['r','relu'],
    'tanh':['t'],
    'leaky_ReLU':['lr','leakyrelu','lrelu'],
    'DReLU':['dr','drelu'],
    'softplus':['sp']
    }

acnames={alias:acname for acname,aliases in ac_aliases.items() for alias in aliases+[acname]}

activations={alias:globals()[acname] for alias,acname in acnames.items()}



@jax.jit
def sqlossindividual(Y1,Y2):
    Y1,Y2=[jnp.squeeze(_) for _ in (Y1,Y2)]
    return jnp.square(Y1-Y2)


@jax.jit
def norm(Y):
    return jnp.sqrt(sqloss(0,Y))


@jax.jit
def relloss(Y1,Y2):
    return sqloss(Y1,Y2)/sqloss(0,Y2)


@jax.jit
def dot_nd(A,B):
    return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))



@jax.jit
def collapselast(A,k):
    dims=A.shape
    #return collapse(A,dims-k,dims)
    return jnp.reshape(A,dims[:-2]+(dims[-2]*dims[-1],))


def randperm(*Xs):
    X=Xs[0]
    n=X.shape[0]
    p=np.random.permutation(n)
    PXs=[np.array(X)[p] for X in Xs]
    #return [jnp.stack([Y[p_i] for p_i in p]) for Y in args]
    return [jnp.array(PX) for PX in PXs]
    

@jax.jit
def apply_on_n(A,X):

    _=jnp.dot(A,X)
    out= jnp.swapaxes(_,len(A.shape)-2,-2)

    return out


@jax.jit
def flatten_first(X):
    blocksize=X.shape[0]*X.shape[1]
    shape=X.shape[2:]
    return jnp.reshape(X,(blocksize,)+shape)
    

    


@jax.jit
def allmatrixproducts(As,Bs):
    products=apply_on_n(As,Bs)
    return flatten_first(products)


def scale(f,C):
    #return jax.jit(lambda X:C*f(X))
    return lambda X:C*f(X)


def normalize(f,X_,echo=False):

    scalesquared=sqloss(f(X_),0)
    C=1/math.sqrt(scalesquared)
    if echo:
        tracking.log('normalized by factor {:.3}'.format(C))
    return scale(f,C)


def normalize_by_weights(learner,X_):
    f=learner.as_static()    
    scalesquared=sqloss(f(X_),0)
    C=1/math.sqrt(scalesquared)

    weights=learner.weights
    weights[0][-1]=weights[0][-1]*C



def closest_multiple(f,X,Y_target):
    Y=f(X)
    C=jnp.dot(Y,Y_target)/jnp.dot(Y,Y)
    return scale(f,C)





def chop(*Xs,blocksize):
    S=Xs[0].shape[0]
    limits=[(a,min(a+blocksize,S)) for a in range(0,S,blocksize)]
    return [tuple([X[a:b] for X in Xs]) for a,b in limits]
    


def takesparams(f):
    return len(signature(f).parameters)==2    

def pad(f):
    return f if takesparams(f) else dummyparams(f)

def fixed_f_Op(A):
    return lambda f: noparams(A(pad(f))) 



def eval_blockwise(f,params,X,blocksize=100000,msg=None):
    f=pad(f)
    _,n,_=X.shape    
    Xs=chop(X,blocksize=blocksize)
    out=[]
    for i,(B,) in enumerate(Xs):
        out.append(jnp.squeeze(f(params,B)))
        #if msg!=None and len(Xs)>1:
            #tracking.trackcurrenttask(msg,(i+1)/len(Xs))
    return jnp.concatenate(out,axis=0)


def blockwise_eval(fdescr,**kw):
    def b_eval(X):
        return eval_blockwise(fdescr.eval,fdescr.weights,X,**kw)
    return b_eval

#def makeblockwise(f):
#    if takesparams(f):
#        blockwise=lambda params,X,**kw: eval_blockwise(f,params,X,**kw)
#    else:
#        blockwise=lambda X,**kw: eval_blockwise(f,None,X,**kw)
#    return blockwise



def leafwise(op,*Gs):
    G1=Gs[0]
    if G1 is None:
        return None
    elif type(G1)==list or type(G1)==tuple:
        return [leafwise(op,*gs) for gs in zip(*Gs)]
    else:
        return op(*Gs)

#def addgrads(G1,G2):
#    if G1==None:
#        return G2
#    elif type(G2)==list or type(G2)==tuple:
#        return [addgrads(g1,g2) for g1,g2 in zip(G1,G2)]
#    else:
#        return G1+G2
def addgrads(G1,G2):
    return leafwise(jnp.add,G1,G2)
        
def scalegrad(G,r):
    if type(G)==list:
        return [scalegrad(g,r) for g in G]
    else:
        return r*G


def sumgrads(Gs):
    Gsum=None
    for G in Gs:
        Gsum=addgrads(Gsum,G)
    return Gsum

def avg_grads(Gs):
    Gsum=None
    for G in Gs:
        Gsum=addgrads(Gsum,G)
    return scalegrad(Gsum,1/len(Gs))



def distinguishable(x,y,p_val=.10,**kwargs): # alternative='greater' to stop when no longer decreasing
    u,p=st.mannwhitneyu(x,y,**kwargs)
    return p<p_val


def substringslast(_strings_):
    strings=deque([s for s in _strings_])
    out=[]
    while len(strings)>0:
        a=strings.pop()
        if all([a not in b for b in strings if a!=b]):
            out.append(a)
        else:
            strings.appendleft(a)
    return out



def donothing(*args):
    pass


def fixparams(f_,params):

    @jax.jit
    def f(X):
        return f_(params,X)
    return f


def noparams(f_):
    return fixparams(f_,None)


def dummyparams(f):
    @jax.jit
    def f_(_,x):
        return f(x)
    return f_


def keyfromstr(s):
    return rnd.PRNGKey(hash(s))



def applyonleaves(T,fn):
    if T is None:
        return None
    elif type(T)==list or type(T)==tuple:
        return [applyonleaves(e,fn) for e in T]
    else:
        return fn(T)

def recurseonleaves(T,leaf_fn,combine):
    if type(T)==list or type(T)==tuple:
        return combine([recurseonleaves(t,leaf_fn,combine) for t in T])
    else:
        return leaf_fn(T)

nestedstructure=applyonleaves


def dimlist(T):
    return nestedstructure(T,lambda A:A.shape if isinstance(A,jnp.ndarray) else str(type(A)))



def applyalonglast(f,X,last):
    lshape,rshape=X.shape[:-last],X.shape[-last:]
    batchsize=np.product(lshape)
    X_=jnp.reshape(X,(batchsize,)+rshape)
    Y_=f(X_)
    return jnp.squeeze(jnp.reshape(Y_,lshape+(-1,)))




def appendtoeach(listdict,elementdict):
    for name,val in elementdict.items():
        if name not in listdict.keys(): listdict[name]=[]
        listdict[name].append(val)


def trycomp(fn,*args):
    try: return fn(*args)
    except: return None

#    if type(l)==list:
#        return [dimlist(e) for e in l]
#    else:
#        return l.shape
    
def shapestr(l):
    return str(dimlist(l))


def printshape(l,msg=''):
    tracking.log(msg+shapestr(l))


def scalarfunction(f):
    def g(*inputs):
        return jnp.squeeze(f(*inputs))
    return g


def combinelossgradfns(lossgradfns,nums_inputs,coefficients):
    #@jax.jit
    def combinedlossgradfn(params,X,*Ys):
        losses,grads=zip(*[lossgrad(params,X,*Ys[:numinputs-1]) for lossgrad,numinputs in zip(lossgradfns,nums_inputs)])
        
        total_loss=sum([loss*c for loss,c in zip(losses,coefficients)])
        total_grad=sumgrads([scalegrad(grad,c) for grad,c in zip(grads,coefficients)])
        return total_loss,total_grad

    return combinedlossgradfn




#def deltasquared(w):
#    sqdists=jnp.sum(jnp.square(w[:,None,:]-w[None,:,:]),axis=-1)
#    return 1/jnp.max(jnp.triu(1/sqdists))

def initweights(shape):
    return rnd.normal(tracking.nextkey(),shape)*jnp.sqrt(cfg.initweight_coefficient/shape[-1])





def squarefn(f):
    return jax.jit(lambda X:X**2)


def compose(*functions):

    def composed(params,X):
        for f,param in zip(functions,params):
            X=f(param,X)
        return X

    return jax.jit(composed)

    
def recompose(ffff,h):

    def hffff(params,X):
        Y=ffff(params[:-1],X)
        return h(params[-1],Y)

    return hffff



def forfixedparams(_op_):
    return lambda f: noparams(_op_(dummyparams(f)))

def forcurrentparams(op):
    return lambda f_descr: fixparams(op(f_descr._eval_),f_descr.weights)

def make_single_x(F):
    return lambda *x: jnp.squeeze(F(*x[:-1],jnp.expand_dims(x[-1],axis=0)))