import util
import jax.numpy as jnp


x=jnp.arange(0,10,.01)
y=jnp.square(x)
util.gausssmooth(x,y,variance=1)
