import jax
import jax.numpy as jnp
from cancellations.utilities import numutil



def genlocalkinetic(psi):
    vg=jax.value_and_grad(lambda params,x:jnp.squeeze(psi(params,x)),argnums=1)
    def E_singlesample(params,x):
        val,grad=vg(params,jnp.expand_dims(x,axis=0))
        return jnp.sum(grad**2)/(2*val**2)
    return jax.jit(jax.vmap(E_singlesample,in_axes=(None,0)))

def genlocalenergy(psi,potential):
    K_local=genlocalkinetic(psi)
    return jax.jit(lambda params,X: K_local(params,X)+potential(X))

def samplewise_value_and_grad(_psi_):
    singlesamplepsi=jax.jit(lambda params,X: jnp.squeeze(_psi_(params,jnp.expand_dims(X,axis=0))))
    return jax.vmap(jax.value_and_grad(singlesamplepsi),in_axes=(None,0))

def samplewise_grad(_psi_):
    singlesamplepsi=jax.jit(lambda params,X: jnp.squeeze(_psi_(params,jnp.expand_dims(X,axis=0))))
    return jax.vmap(jax.grad(singlesamplepsi),in_axes=(None,0))

class Energy_val_and_grad:
    def __init__(self,psi_descr,potential):
        psi=psi_descr._eval_
        self.le=genlocalenergy(psi,potential)
        self.Lq=lambda params,X: jnp.log(psi(params,X)**2)
        self.DLq=samplewise_grad(self.Lq)

    def _eval_(self,params,X,*a):
        LE=self.le(params,X)
        L=jnp.average(LE)
        coefficients=(LE-L)/len(LE)
        DLq=self.DLq(params,X)
        grad=numutil.applyonleaves(DLq,lambda T:jnp.tensordot(coefficients,T,axes=(0,0)))
        return L,grad
