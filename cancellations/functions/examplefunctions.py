import jax, jax.numpy as jnp
import itertools
from cancellations.functions import _functions_
from cancellations.utilities import numutil as mathutil
from cancellations.tracking import tracking
from cancellations.tracking.tracking import dotdict
import numpy as np, math
from jax.lax import complex



#----------------------------------------------------------------------------------------------------
# polynomials
#----------------------------------------------------------------------------------------------------

def monomials(x,n):
    xk=jnp.ones_like(x)
    out=[]
    for k in range(n+1):
        out.append(xk)    
        xk=x*xk
    return jnp.stack(out,axis=-1)

def polynomial(coefficients,X):
    n=len(coefficients)-1
    return jnp.inner(monomials(X,n),coefficients)
        
    
#----------------------------------------------------------------------------------------------------
# Hermite polynomials
#----------------------------------------------------------------------------------------------------

            
def H_coefficients(n):
    if n==0:
        return [[1]]
    else:
        A=H_coefficients(n-1)
        a1=A[-1]+2*[0]
        a=[-a1[1]]
        for k in range(1,n+1):
            a.append(2*a1[k-1]-(k+1)*a1[k+1])
        A.append(a)
        return A


H_coefficients_list=[jnp.array(p) for p in H_coefficients(25)]


#----------------------------------------------------------------------------------------------------
# H_O_solution with -h-,m,k=1
#----------------------------------------------------------------------------------------------------


def psi(n):
    assert(n>0)
    p=mathutil.fixparams(polynomial,H_coefficients_list[n-1])
    psi_n=jax.jit(lambda x: jnp.exp(-x**2/2)*p(x))
    return psi_n

def totalenergy(n): return sum([i+1/2 for i in range(n)])



# load function definitions

# for i in range(1,11):
#     fname='psi'+str(i)
#     setattr(functions,fname,psi(i))
#     #globals()[fname]=psi(i)

_functions_.square=lambda y:y**2




#----------------------------------------------------------------------------------------------------
# for d>1
# f1,..,fn need only be pairwise different in one space dimension
#----------------------------------------------------------------------------------------------------

# S+k-1 choose k-1
def sumsto(k,S):
    return [[b-a-1 for a,b in zip((-1,)+t,t+(S+k-1,))] for t in itertools.combinations(range(S+k-1),k-1)]


def gen_n_dtuples(n,d):
    s=0
    out=[]
    while len(out)<=n:
        out=out+sumsto(d,s)
        s+=1
        
    return out[:n]

def n_dtuples_maxdegree(n,d):
    return max([max(t) for t in gen_n_dtuples(n,d)])


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

psis=[psi(i) for i in range(1,25)]

def genpsi(d,ijk):
    #psis=[getattr(examplefunctions,'psi{}'.format(i)) for i in range(1,11)]
    def psi_ijk(X):
        out=1
        for k,l in zip(ijk,range(d)):
            out*=psis[k](X[:,l])
        return out
    return psi_ijk

for d in [1,2,3]:
    for i,ijk in enumerate(gen_n_dtuples(30,d)):
        psi=genpsi(d,ijk)
        setattr(_functions_,'psi{}_{}d'.format(i+1,d),psi)







#----------------------------------------------------------------------------------------------------


def get_harmonic_oscillator2d(n,d,upshift=0,imax=None):
    if imax is not None:
        assert(imax>=n)
        upshift=imax-n
    I=list(np.arange(1,n+1)+upshift)
    return _functions_.Slater(*['psi{}_{}d'.format(i,d) for i in I])

def getlearner_example_profile(n,d):
    learnerparams=tracking.dotdict(\
        SPNN=dotdict(widths=[d,100,50],activation='sp'),\
        backflow=dotdict(widths=[50,50],activation='sp'),\
        dets=dotdict(d=50,ndets=10),)
        #'OddNN':dict(widths=[25,1],activation='sp')
    return learnerparams

def getlearner_example(n,d,profile=None):
    if profile is None: profile=getlearner_example_profile(n,d)
    return _functions_.Product(_functions_.IsoGaussian(1.0),_functions_.ComposedFunction(\
        _functions_.SingleparticleNN(**profile['SPNN']),\
        _functions_.Backflow(**profile['backflow']),\
        _functions_.Dets(n=n,**profile['dets']),\
        _functions_.Sum()\
        ))


#----------------------------------------------------------------------------------------------------

class ExpSlater(_functions_.FunctionDescription):

    @staticmethod
    def _initweights_(m,n,d,**kw):
        return _functions_.Barron._initweights_(m,n,d,**kw)

    def compile(self):
        def f(weights,X):
            (Ws,bs),a=weights
            Ms=jnp.exp(complex(0.0,1.0)*(jnp.swapaxes(jnp.inner(X,Ws),1,2)+bs[None,:,None,None]))
            out=jnp.dot(jnp.real(jnp.linalg.det(Ms)),jnp.squeeze(a))

            n,d=X.shape[-2:]
            return out/jnp.sqrt(math.factorial(n))
        return f

    

class QuadrantSlater(_functions_.FunctionDescription):
    def compile(self):
        if self.d==1:
            signs=[[1],[-1]]
        if self.d==2:
            signs=[[1,1],[1,-1],[-1,1],[-1,-1]]
        if self.d==3:
            signs=[[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],\
                  [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]
        signs=jnp.array(signs)
        def f(t,X):
            Ms=jax.nn.sigmoid(t*jnp.inner(X,signs[:self.n,:]))
            out=jnp.linalg.det(Ms)

            n,d=X.shape[-2:]
            return out/jnp.sqrt(math.factorial(n))
        return f

    @staticmethod
    def _initweights_(**kw):
        if 'w' in kw: return kw['w']
        return 1.0



