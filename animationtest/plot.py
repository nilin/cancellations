import matplotlib.pyplot as plt
import jax.numpy as jnp

x=jnp.arange(-1,1,.01)

#fig=plt.figure()


for i in range(1000):
	y=jnp.sin(x*i*.1)
	plt.cla()
	plt.plot(x,y)
	plt.pause(0.01)
