import jax
import jax.numpy as jnp
import jax.random as rnd
from functools import partial
from jax.tree_util import tree_map
from cancellations.tracking import tracking
from cancellations.functions import _functions_
from cancellations.utilities import numutil
from cancellations.utilities.numutil import make_single_x




@jax.jit
def weighted_SI_loss(Y,Y_target,relweights):
    overlap=lambda Y1,Y2,weights: jnp.average(Y1*Y2*weights)
    return 1-overlap(Y,Y_target,relweights)**2/(overlap(Y,Y,relweights)*overlap(Y_target,Y_target,relweights))

####################################################################################################

def get_lossgrad_SI(f):
    lossfn=lambda params,X,Y,rho: weighted_SI_loss(f(params,X),Y,relweights=1/rho)
    return jax.jit(jax.value_and_grad(lossfn))

def get_lossgrad_NONSI(f):
    lossfn=lambda params,X,Y,rho: jnp.average((f(params,X)-Y)**2/rho)   #(f(params,X),Y,relweights=1/rho)
    return jax.jit(jax.value_and_grad(lossfn))

@jax.jit
def norm(Y,rho):
    return jnp.sqrt(jnp.average(Y**2/rho))


####################################################################################################

def transform_grad(lossgrad,T,fromrawloss=False):

    if fromrawloss: lossgrad=jax.value_and_grad(lossgrad)

    def newlossgrad(params,*X):
        val,grad=lossgrad(params,*X)
        dT=jax.grad(T)(val)
        return val,tree_map(lambda A:dT*A,grad)

    return jax.jit(newlossgrad)

