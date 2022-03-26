import math
import util
import jax
import jax.numpy as jnp
import bookkeep as bk
import matplotlib.pyplot as plt

key=jax.random.PRNGKey(0)
x_range,y_=bk.getdata('gamma_ReLU')['vals']
f=util.listasfunction(x_range,y_)

samples=1000
r=1
x=jnp.tanh(r*jax.random.normal(key,(samples,1)))
y=f(x)

#Y=jnp.concatenate([jnp.ones((samples,1)),x,jnp.square(x),jnp.exp(x)],axis=-1)
Y=jnp.concatenate([jnp.ones((samples,1)),x,jnp.square(x)],axis=-1)

a,dist=util.basisfit(y,Y)

I_=jnp.arange(-1,1.01,.01)

print(a)
plt.plot(I_,f(I_),'b')
#plt.plot(x_,a[3]*jnp.exp(x_)+a[2]*jnp.square(x_)+a[1]*x_+a[0],'r--')
#plt.plot(x_,a[2]*jnp.square(x_)+a[1]*x_+a[0],'r--')


def g(x):
	par=jnp.multiply(x,jnp.arctan(jnp.sqrt((x+1)/(-x+1))))+jnp.sqrt(-jnp.square(x)+1)/2
	return (1/math.pi)*par

plt.plot(I_,g(I_),'m')
plt.savefig('plots/gamma ReLU.pdf')
plt.show()
