import jax, jax.numpy as jnp
import jax.random as rnd

from cancellations.utilities import numutil, tracking,config as cfg
from cancellations.utilities.numutil import activations
from cancellations.utilities import numutil
from cancellations.functions import permutations as ps


#=======================================================================================================
# NN 
#=======================================================================================================

def gen_NN_layer(ac):
    activation=activations[ac]

    if cfg.layernormalization is None:
        layernormalize=lambda Y:Y
    else:
        std,mode=cfg.layernormalization

        match mode:
            case 'online':
                def layernormalize(Y):
                    means=jnp.average(Y,axis=-1)
                    norms=jnp.sqrt(jnp.average(Y**2,axis=-1))
                    return std*(Y-means[:,None])/norms[:,None]
            case 'batch':
                def layernormalize(Y):
                    mean=jnp.average(Y)
                    norm=jnp.sqrt(jnp.average(Y**2))
                    return std*(Y-mean)/norm

    @jax.jit
    def f(Wb,X):
        W,b=Wb[0],Wb[1]
        ac_inputs=jnp.inner(X,W)+b[None,:]
        ac_inputs=layernormalize(ac_inputs)
        return activation(ac_inputs)
    return f
        

def gen_NN_wideoutput(ac):
    L=gen_NN_layer(ac)

    @jax.jit
    def NN(params,X):
        for Wb in params:
            X=L(Wb,X)
        return X

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
# single layer
#=======================================================================================================

#def gen_singlelayer_Af(n,ac):
#
#    Ps,signs=ps.allperms(n)                    # Ps:    n!,n,n
#
#    @jax.jit
#    def Af_singleneuron(w,b,X):
#        overlaps=jnp.inner(X,w)
#
#
#    #    PX=mathutil.apply_on_n(Ps,X)                # PX:    n!,s,n,d
#    #    fX=f(params,PX)                        # fX:    n!,s
#    #    return jnp.dot(signs,fX)                # s
#
#    return Af






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



