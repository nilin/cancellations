import jax, jax.random as rnd, jax.numpy as jnp
from cancellations.config import config as cfg, tracking
from cancellations.utilities import numutil
from cancellations.utilities import permutations as ps
from cancellations.functions import NN as mv




####################################################################################################
#
# backflow
#
# general equivariant function
#
####################################################################################################


"""
# phi: params x R^d x R^d -> R^d'
# 
# output F
#
# F: params x R^nd x R^md -> R^nd'
#
# F(W,X,Y) equivariant in X, symmetric in Y
"""
def gen_EV_layer(phi,pool=jnp.sum):
    
    phi_iJ=jax.vmap(phi,in_axes=(None,None,-2),out_axes=-1)
    def pooled_along_y(params,xi,yJ):
        return pool(phi_iJ(params,xi,yJ),axis=-1)

    return jax.jit(jax.vmap(pooled_along_y,in_axes=(None,-2,None),out_axes=-2))


"""
# F(params,PX)=P' F(params,X),
#
# where P' applies P on dimension -2
"""
def gen_backflow(activation):

    ac=numutil.activations[activation]
    phi=jax.jit(lambda Wb,x,y:ac(mv.NN_layer(Wb,jnp.concatenate([x,y],axis=-1))))
    layer=gen_EV_layer(phi)
    
    def F(params,Y):
        for Wl in params:
            Y=layer(Wl,Y,Y)    
        return Y
    return jax.jit(F)

def gen_singleparticleNN(activation):
    return jax.vmap(mv.gen_NN_wideoutput(activation),in_axes=(None,-2),out_axes=-2)

def initweights_Backflow(widths,*args,**kw):
    ds=widths
    Ws=[numutil.initweights((d2,2*d1)) for d1,d2 in zip(ds[:-1],ds[1:])]
    bs=[rnd.normal(tracking.nextkey(),(d2,))*cfg.biasinitsize for d2 in ds[1:]]

    return list(zip(Ws,bs))    




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
        PX=numutil.apply_on_n(Ps,X)                # PX:    n!,s,n,d
        fX=f(params,PX)                        # fX:    n!,s
        return jnp.tensordot(signs,fX,axes=(0,0))                # s

    return Af


def gen_Af(n,f):
    return gen_Af_general(n,f)
    
        
#=======================================================================================================
# single layer special case
#=======================================================================================================

def gen_singlelayer_Af(n,d,activation,compatibilitymode=False):
    ac=numutil.activations[activation]

    Ps,signs=ps.allpermtuples(n)                    # Ps:    n!,n,n
    I=jnp.repeat(jnp.expand_dims(jnp.arange(n),axis=0),len(signs),axis=0)
    scale=1/jnp.sqrt(len(signs))

    @jax.jit
    def Af_singleneuron(w,b,X):     # w: n,d
        overlaps=jnp.inner(X,w)     # s,n,n
        outputs=ac(jnp.sum(overlaps[...,I,Ps],axis=-1)+b)
        return jnp.inner(outputs,signs)

    @jax.jit
    def Af(params,X):     # w: n,d
        (W,b),a=params
        AS_neuronoutputs=jax.vmap(Af_singleneuron,in_axes=(0,0,None),out_axes=-1)(W,b,X)
        return jnp.squeeze(jnp.dot(AS_neuronoutputs,a)*scale)

    return Af

#    @jax.jit
#    def Af_(params,X):
#        (W,b),(A,_)=params
#        W_=jnp.reshape(W,(-1,n,d))
#        print(W_)
#        print(b)
#        print(A)
#        return Af([(W_,b),A],X)
#
#    return Af_ if compatibilitymode else Af



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

def diagprods(A):
    return jnp.product(jnp.diagonal(A,axis1=-2,axis2=-1),axis=-1)

def prods(A,Y):
    snkn=jnp.inner(Y,A)
    sknn=jnp.swapaxes(snkn,-3,-2)
    return diagprods(sknn)


