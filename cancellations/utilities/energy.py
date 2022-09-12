import jax
import jax.numpy as jnp




def genlocalkinetic(psi):
    vg=jax.value_and_grad(lambda params,x:jnp.squeeze(psi(params,x)),argnums=1)

    def E_singlesample(params,x):
        #val,grad=vg(params,x)
        val,grad=vg(params,jnp.expand_dims(x,axis=0))
        return jnp.sum(grad**2)/(2*val**2)

    return jax.jit(jax.vmap(E_singlesample,in_axes=(None,0)))

