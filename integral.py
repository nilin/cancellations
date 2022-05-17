import jax
import bookkeep as bk
import math
import jax.numpy as jnp
from scipy.io import loadmat
import matplotlib.pyplot as plt

Fs={'ReLU':lambda p:1/(jnp.sqrt(2*math.pi)*jnp.square(p)),'HS':lambda p:1/(jnp.sqrt(2*math.pi)*p),'tanh':lambda p:jnp.sqrt(math.pi/2)/jnp.sinh(math.pi*p/2)}

#
#def array_to_function(y,f_):
#	std=y[1]-y[0]
#	def f(x):
#		weights=jnp.exp(-jnp.square(x[:,None]-y[None,:])/(2*std**2))
#		return jnp.dot(weights,f_)/jnp.sum(weights,axis=-1)
#	return f
#
#		
#def array_to_function_logscale(t,f_):
#	y=jnp.log(t)
#	f_exp=array_to_function(y,f_)
#	def f(s):
#		x=jnp.log(s)
#		return f_exp(x)
#	return f

def integral(t,diag,F):
	x=jnp.log(t)
	dx=x[1]-x[0]
	return 1/jnp.sqrt(2*math.pi)*jnp.sum(diag*jnp.square(F(t))*t*dx)


def getdata_and_integrate(n,i,ac):
	data=loadmat('matlab/Ds/n='+str(n)+'/instance='+str(i))
	w=jnp.squeeze(data['W'])
	thetas=jnp.squeeze(data['ts'])
	diag=jnp.squeeze(data['diag'])

	mu=jnp.sqrt(jnp.sum(jnp.square(w)))
	if ac != 'exp':	
		return integral(thetas,diag,Fs[ac])*2/mu
	else:
		return jnp.exp(2*mu*thetas)*diag



for n in range(2,26):
	for ac in ['ReLU','HS','tanh','exp']:
		bk.save(jnp.array([getdata_and_integrate(n,i,ac) for i in range(1,101)]),'computed_by_integral/'+ac+' n='+str(n))
