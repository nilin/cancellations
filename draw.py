import matplotlib.pyplot as plt
import jax.numpy as jnp

x_=jnp.arange(-1,1,.01)
plt.plot(x_,jnp.arctan(jnp.sqrt((x_+1)/(-x_+1))))
plt.show()
