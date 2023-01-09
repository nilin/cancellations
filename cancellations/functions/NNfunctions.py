import jax, jax.numpy as jnp
import jax.random as rnd
import math

from cancellations.utilities import numutil, tracking,config as cfg
from cancellations.utilities.numutil import activations
from cancellations.utilities import numutil
from cancellations.utilities import permutations as ps


#=======================================================================================================
# NN 
#=======================================================================================================

@jax.jit
def NN_layer(Wb,X):
    W,b=Wb[0],Wb[1]
    return jnp.inner(X,W)+b[None,:]

def gen_NN_wideoutput(activation):
    ac=activations[activation]
    @jax.jit
    def NN(params,X):
        for Wb in params[:-1]:
            X=ac(NN_layer(Wb,X))
        return NN_layer(params[-1],X)
    return NN

def gen_NN(activation):
    return numutil.scalarfunction(gen_NN_wideoutput(activation))

def gen_NN_NS(activation):
    NN=gen_NN(activation)

    @jax.jit
    def NN_NS(params,X):
        X=numutil.collapselast(X,2)
        return NN(params,X)

    return NN_NS

def gen_lossgrad(f,lossfn=None):
    if lossfn is None: lossfn=cfg.getlossfn()

    def collectiveloss(params,X,*Y):
        return lossfn(f(params,X),*Y)

    return jax.value_and_grad(collectiveloss)
    

#=======================================================================================================
# single layer special case
#=======================================================================================================

def gen_singlelayer_Af(n,ac):

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
        (W,bs),a=params
        A_neuronoutputs=jax.vmap(Af_singleneuron,in_axes=(0,0,None),out_axes=-1)(W,bs,X)
        return jnp.squeeze(jnp.inner(A_neuronoutputs,a))*scale

    return Af


#----------------------------------------------------------------------------------------------------
# random initializations
#----------------------------------------------------------------------------------------------------


def initweights_NN(widths,*args,**kw):
    ds=widths
    Ws=[numutil.initweights((d2,d1)) for d1,d2 in zip(ds[:-1],ds[1:])]
    bs=[rnd.normal(tracking.nextkey(),(d2,))*cfg.biasinitsize for d2 in ds[1:]]

    return list(zip(Ws,bs))


#----------------------------------------------------------------------------------------------------
# operations on functions
#----------------------------------------------------------------------------------------------------

def multiply(*fs):
    if max([numutil.takesparams(f) for f in fs]):
        def F(paramsbundle,X):
            out=1
            for f,params in zip(fs,paramsbundle):
                out*=numutil.pad(f)(params,X)
            return out
    else:
        def F(X):
            out=1
            for f in fs: out*=f(X)
            return out
    return F




def inspect_composition(steps,params,X):
    layers=[(None,X)]
    for step,weights in zip(steps,params):
        try:
            layers.append((weights,step(weights,layers[-1][-1])))
        except Exception as e:
            layers.append((weights,None))
    return layers



