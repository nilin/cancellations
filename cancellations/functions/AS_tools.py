#
# nilin
#
# 2022/7
#


import jax, jax.numpy as jnp

from cancellations.utilities import numutil as mathutil
from cancellations.utilities import permutations as ps
from cancellations.functions import NNfunctions as mv




#=======================================================================================================
#
# explicit AS
#
# basic AS (n=1,..,8)
#
#=======================================================================================================

def gen_Af_general(n,f):
    Ps,signs=ps.allperms(n)                    # Ps:    n!,n,n

    @jax.jit
    def Af(params,X):
        PX=mathutil.apply_on_n(Ps,X)                # PX:    n!,s,n,d
        fX=f(params,PX)                        # fX:    n!,s
        return jnp.dot(signs,fX)                # s

    return Af



#----------------------------------------------------------------------------------------------------

def gen_Af(n,f):
    return gen_Af_general(n,f)
    


#def gen_lossgrad_Af(n,f,lossfn):
#    return mv.gen_lossgrad(gen_Af(n,f)) #if n<=cfg.heavy_threshold else AS_HEAVY.gen_lossgrad_Af_heavy(n,f,lossfn)

        

#=======================================================================================================
#
# backflow+det
#
#=======================================================================================================


@jax.jit
def dets(A,Y):
    snkn=jnp.inner(Y,A)
    sknn=jnp.swapaxes(snkn,-3,-2)
    return jnp.linalg.det(sknn)


def inspectdetsum(A,Y):
    def step1(A,Y):
        snkn=jnp.inner(Y,A)
        return jnp.swapaxes(snkn,-3,-2)
    def step2(_,Y): return jnp.linalg.det(Y)
    def step3(_,Y): return jnp.sum(Y,axis=-1)
    return mv.inspect_composition([step1,step2,step3],[A,None,None],Y)

#=======================================================================================================
def diagprods(A):
    return jnp.product(jnp.diagonal(A,axis1=-2,axis2=-1),axis=-1)

def prods(A,Y):
    snkn=jnp.inner(Y,A)
    sknn=jnp.swapaxes(snkn,-3,-2)
    return diagprods(sknn)


#=======================================================================================================



#=======================================================================================================
## test
#=======================================================================================================
#
#def test_AS(Ws,bs,X):
#
#    Af=lambda x:AS_NN(Ws,bs,x)
#    f=lambda x:NN(Ws,bs,x)
#
#    testing.test_AS(Af,f,X)
#
#
#if __name__=='__main__':
#    pass    
#