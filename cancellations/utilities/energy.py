import jax
import jax.numpy as jnp
from ..utilities import numutil, textutil, tracking






def genlocalkinetic(psi):
    vg=jax.value_and_grad(lambda params,x:jnp.squeeze(psi(params,x)),argnums=1)

    def E_singlesample(params,x):
        #val,grad=vg(params,x)
        val,grad=vg(params,jnp.expand_dims(x,axis=0))
        return jnp.sum(grad**2)/(2*val**2)

    return jax.jit(jax.vmap(E_singlesample,in_axes=(None,0)))



def genlocalenergy(psi,potential):
    K_local=genlocalkinetic(psi)
    return jax.jit(lambda params,X: K_local(params,X)+potential(X))


#
#def gen_logenergy_grad(localenergy,_p_):
#
#    S1=lambda params,X: jnp.sum(localenergy(params,X))
#    _d1n11_=jax.value_and_grad(S1)
#
#    _logp_=lambda params,X: jnp.log(_p_(params,X))
#    _Dlogp_=samplewise_value_and_grad(_logp_)
#
#    def _n12_(params,X):
#        Dlogp_X=_Dlogp_(params,X)
#        L_X=localenergy(params,X)
#        return numutil.applyonleaves(Dlogp_X,lambda T:jnp.tensordot(L_X,T))
#
#    S2=lambda params,X:_logp_(params,X)
#    _n2_=jax.grad(S2)
#
#    def gradlog(params,X):
#
#        # debug
#        localenergy1=localenergy
#        _Dlogp_1=_Dlogp_
#        _logp_1=_logp_
#        _p_1=_p_
#        #
#
#        d1,n11=_d1n11_(params,X)
#        A=(n11+_n12_(params,X))/d1
#        B=_n2_(params,X)/X.shape(0)
#        return A-B
#
#    return jax.jit(gradlog)
#


def gen_logenergy_grad(localenergy,_p_):

    _logp_=lambda params,X: jnp.log(_p_(params,X))
    #_logp_Dlogp_=samplewise_grad(_logp_)


    def gradlog(params,X):

        S1=lambda params,X: jnp.sum(localenergy(params,X))
        d1,n11=jax.value_and_grad(S1)(params,X)

        Dlogp_X=samplewise_grad(_logp_)(params,X)
        L_X=localenergy(params,X)
        n12=numutil.applyonleaves(Dlogp_X,lambda T:jnp.tensordot(L_X,T,axes=(0,0)))

        S2=lambda params,X:jnp.sum(_logp_(params,X))
        n2=jax.grad(S2)(params,X)

        return numutil.leafwise(lambda A,B,C: (A+B)/d1-C/X.shape[0], n11,n12,n2)

    return jax.jit(gradlog)


#def gen_logenergy_grad(localenergy,_p_):
#
#    def gradlog(params,X):
#


    # numerator:    I L(params,X)*p(params,X) dx
    # denominator:  I p(params,X) dx

    # D-numerator:  I DL*p dx + I L*Dp/p(params,X) pdx
    #           ~   sum DL(params,X) + sum L(params,X)*Dp/p(params,X)
    #           =   sum DL(params,X) + sum L(params,X)*D(log p)(params,X)

    # D(log numerator) ~ [sum DL + sum L*D(log p)] / sum L

    # D-denominator: I Dp dx = I Dp/p pdx = I D(log p) pdx ~ sum D(log p)
    # D(log denominator) ~ sum D(log p) / sum 1


def samplewise_value_and_grad(_psi_):
    singlesamplepsi=jax.jit(lambda params,X: jnp.squeeze(_psi_(params,jnp.expand_dims(X,axis=0))))
    return jax.vmap(jax.value_and_grad(singlesamplepsi),in_axes=(None,0))

def samplewise_grad(_psi_):
    singlesamplepsi=jax.jit(lambda params,X: jnp.squeeze(_psi_(params,jnp.expand_dims(X,axis=0))))
    return jax.vmap(jax.grad(singlesamplepsi),in_axes=(None,0))