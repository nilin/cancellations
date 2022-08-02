import jax
import jax.numpy as jnp



@jax.jit
def ptanh(X):
	return jnp.maximum(jnp.tanh(X),0)


c_acs={'ptanh':ptanh}
