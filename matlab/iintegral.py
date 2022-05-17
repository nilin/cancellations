import jax
import bookkeep as bk
import math
import jax.numpy as jnp
from scipy.io import loadmat
import matplotlib.pyplot as plt


F_ReLU=lambda p:1/(jnp.sqrt(2*math.pi)*jnp.square(p))


def array_to_function(y,f_):
	std=y[1]-y[0]
	def f(x):
		weights=jnp.exp(-jnp.square(x[:,:,None]-y[None,None,:])/(2*std**2))
		return jnp.dot(weights,f_)/jnp.sum(weights,axis=-1)
	return f

		
def array_to_function_logscale(t,f_):
	y=jnp.log(t)
	f_exp=array_to_function(y,f_)
	def f(s):
		x=jnp.log(s)
		return f_exp(x)
	return f

		
def extend_to_grid(diag,grid,mu):
	s,t=jnp.meshgrid(grid,grid)
	return jnp.exp(-mu*jnp.square(s-t))*diag(jnp.sqrt(s*t)),s,t
	
def quadrantintegral(diag,grid,mu,F,sign):
	D,s,t=extend_to_grid(diag,grid,mu)
	ds=grid[1]-grid[0]

	integrand=D*F(s)*F(t*sign)/(2*math.pi)

	print(s)
	print(integrand)

	return jnp.sum(ds**2*jnp.sum(integrand))
	



#def quadrant(w,t_,diag,F,sign):
#
#	mu=jnp.sum(jnp.square(w))/2
#
#	x=jnp.log(t_)
#	y=x
#
#	dx=x[1]-x[0]
#	dy=y[1]-y[0]
#
#	xpy=x[:,None]+y[None,:]
#	xmy=x[:,None]-y[None,:]
#
#	S=jnp.exp(xpy/2)
#	T=jnp.exp(xmy/2)
#
#	integrand=1/(4*math.pi)*F(S)*F(T*sign)*jnp.exp(-mu*jnp.square(S-T))*diag*t_
#
#	print(dx)
#
#	return jnp.sum(integrand)*dx*dy
#
#
#def integral(w,t_,diag,offdiag):
#	return 2*quadrantReLU(w,t_,diag,1)+2*quadrantReLU(w,t_,offdiag,-1)



def getdata_and_integrate(n,i):
	data=loadmat('matlab/Ds/n='+str(n)+'/instance='+str(i))
	w=jnp.squeeze(data['W'])
	ts=jnp.squeeze(data['ts'])
	diag_=jnp.squeeze(data['diag'])
	offdiag_=jnp.squeeze(data['offdiag'])
	#t_max=min(10*jnp.min(jnp.array(jnp.where(diag_>.9))),100)
	t_max=250

	diag=array_to_function_logscale(ts,diag_)
	offdiag=array_to_function_logscale(ts,offdiag_)

	mu=jnp.sum(jnp.square(w))/2
	thetas=jnp.arange(0,t_max,.25)

	F=F_ReLU
	I=2*quadrantintegral(diag,thetas,mu,F,1)+2*quadrantintegral(offdiag,thetas,mu,F,-1)
	print(I)



for n in range(2,26):
	getdata_and_integrate(n,1)	
	#bk.save([getdata_and_integrate(n,i) for i in range(1,101)],'computed_by_integral/n='+str(n))
